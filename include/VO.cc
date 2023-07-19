

#include "Utils.hpp"
#include "Frame.hpp"
#include "3DHandler.hpp"
#include "MapPoint.hpp"
#include "Camera.hpp"
#include "Features.hpp"
#include "Viz.hpp"
#include "Map.hpp"
#include "OptFlow.hpp"


enum dataset {kitti, tum, euroc};


enum voStatus {INIT, TRACKING, ERROR, RESET};


template <typename T>
class VO {
    private:

        Map::Ptr map;
        Intrinsics::Ptr intrinsics;
        DataHandler::Ptr dataHandler;
        typename Features<T>::Ptr featureDetector;
        // Features<c>::Ptr featureDetector;
        _3DHandler::Ptr Handler3D;
        std::string logsPath;
        bool debugMode;
        voStatus status;
        unsigned long debugSteps;
        Frame::Ptr currFrame;
        Frame::Ptr prevFrame;
        Sophus::SE3d relativeMotion;
        OptFlow::Ptr optFlow;

    public:
        // viewer thread has to be in main thread
        Viewer::Ptr viewer;

        VO( std::string &configFile, dataset datasetType) {
            // initialize glog
            initLogging();

            // initialize the datahandler
            if (datasetType == kitti) {
                LOG(INFO) << "KITTI dataset initialized";
                dataHandler = std::make_shared<KITTI>(configFile);
            } else if (datasetType == tum) {
                LOG(ERROR) << "TUM dataset package not implemented yet";
            } else if (datasetType == euroc) {
                LOG(ERROR) << "EUROC dataset package not implemented yet";
            } else {
                LOG(ERROR) << "Invalid dataset type";
            }
            this->debugMode = dataHandler->getDebugMode();
            if (debugMode) {
                LOG(INFO) << "DEBUG mode enabled";
                this->debugSteps = dataHandler->getDebugSteps();
                LOG(INFO) << "DEBUG steps set to " << debugSteps;

            } else {
                LOG(INFO) << "DEBUG mode disabled";
                this->debugSteps = dataHandler->getTotalFrames();
            }

            // initialize other modules
            initModules();
            // set current and previous frame to nullptr
            currFrame = nullptr;
            prevFrame = nullptr;
            // set status to INIT
            status = INIT;
        }


        void initModules() {
            // Initialize all modules
            // First, check for gpu support on the system
            // If gpu support is available, use templates to initialize the Features and OpticalFlow modules
            // Else, initialize the modules with CPU

            // Initialize the Features module with CUDA
            this->featureDetector = std::make_shared<Features<T>>(DetectorType::ORB, DescriptorType::BRIEF);
            intrinsics = dataHandler->getCalibParams();
            map = std::make_shared<Map>();
            Handler3D = std::make_shared<_3DHandler>(intrinsics);
            map = std::make_shared<Map>();
            viewer = std::make_shared<Viewer>();
            viewer->setMap(map);
            optFlow = std::make_shared<OptFlow>();

        }

        void initLogging() {
            this->logsPath = "logs/";
            if (!boost::filesystem::exists(logsPath)) {
                boost::filesystem::create_directory(logsPath);
            }
            std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
            // convert to string
            std::time_t now_c = std::chrono::system_clock::to_time_t(now);
            std::string now_str = std::ctime(&now_c);
            std::string loggingPath = "logs/" + now_str + "_VO.log";
            google::SetLogDestination(0, loggingPath.c_str());
        }
        bool runVO() {
            while (debugSteps--) {
                std::cout << debugSteps << std::endl;

                if (!takeVOStep()) {
                    LOG(ERROR) << "VO failed at step " << debugSteps;
                    return false;
                    
                }
            }
        }

        bool takeVOStep() {
            Frame::Ptr frame = prepNextFrame();

            if (frame == nullptr) {
                LOG(ERROR) << "No more data to process";
                return false;
            }
            addFrame(frame);
            return true;
        }

        void addFrame(Frame::Ptr frame) {
            currFrame = frame;
            if (status == voStatus::INIT) {
                if (prevFrame == nullptr) {
                    bool success = buildInitMap();
                    if (success) {
                        // update map
                        map->insertKeyFrame(currFrame);
                        map->insertKeyFrame(currFrame->rightFrame);
                        viewer->addCurrentFrame(currFrame);
                        viewer->updateMap();
                        status = voStatus::TRACKING;
                        prevFrame = currFrame;
                    } else {
                        LOG(ERROR) << "Map initialization failed for frame ID: " << currFrame->getFrameID();
                        status = voStatus::INIT;
                    }
                }
            } else if (status == voStatus::TRACKING) {
                // 1. Calculate the approximate pose of this new frame based on relative velocity.
                // 2. reproject 3d points on to the current frame.
                // 3. Calculate the optical flow last frame to current frame 
                // 4. Find how many points are inliers, if less than 20%, register frame as a keyframe and project new points.
                        // Q - Should the outlier be added as new 3d points?

                LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " is a entering TRACKING state";
                if (prevFrame != nullptr) {
                    currFrame->setPose(relativeMotion*prevFrame->getPose());
                }

                // log current frame pose
                LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " has pose: " << currFrame->getPose().matrix3x4();

                // 5. If more than 20% points are inliers, register frame as regular frame, update the mapPoint with new observation and update frame with observation
                int optFlowInlierCount = track();

                //6. If inliers not enough, mask the points that are now features and detect new features that are masked with the already detected features
                relativeMotion = currFrame->getPose()*prevFrame->getPose().inverse();
                map->insertKeyFrame(currFrame);
                map->insertKeyFrame(currFrame->rightFrame);
                viewer->addCurrentFrame(currFrame);
                viewer->updateMap();
                status = voStatus::TRACKING;
                prevFrame = currFrame;

            }


        }

        int track() {
            // reproject all the 3d points seen in previous frame to the current frame

            auto obsMapPoints = prevFrame->getObsMapPoints();
            std::vector<cv::Point2f> prevPts, currPts;
            std::vector<int> mapPointIndex;
            int trackedFeatures = 0;
            for (auto &obsMapPoint : obsMapPoints) {
                cv::Point2f singlePrevPt =  prevFrame->getKeypoints()[obsMapPoint.second->getKpID(prevFrame->getFrameID())].pt;
                
                // now get 3d location of this mapPoint and reproject it to the current frame
                Vec3 mapPoint3D = obsMapPoint.second->getPosition();
                cv::Point2f currPt = currFrame->world2pixel(mapPoint3D, intrinsics);
                // check if the point is within the image

                if (currPt.x < 0 || currPt.y < 0 || currPt.x > currFrame->getRawImg().cols || currPt.y > currFrame->getRawImg().rows) {
                    continue;
                }

                prevPts.push_back(singlePrevPt);
                currPts.push_back(currPt);
                mapPointIndex.push_back(obsMapPoint.first);
                trackedFeatures++;
            }
            // log the number of tracked features for this frame pair
            LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " has " << trackedFeatures << " reprojected 3d points from " << obsMapPoints.size() << " 3d points ";


            int inlierCount = 0;
            // now get the flow between the two frames
            optFlow->getOptFlow(prevFrame, currFrame, prevPts, currPts);
            std::vector<uchar> flowStatus = optFlow->getFlowStatus();
            for (int i=0; i<flowStatus.size();i++) {
                if(flowStatus.at(i)==1) {
                    // Flow is good for this point, add this point as feature to the current frame
                    // first empty the keypoints
                    currFrame->clearKeypoints();
                    currFrame->addKeypoint(cv::KeyPoint(currPts[i], 3));
                    currFrame->addObservation(obsMapPoints[mapPointIndex[i]]);
                    // add this point as observation to the mapPoint
                    obsMapPoints[mapPointIndex[i]]->addObservation(currFrame->getFrameID(), currFrame->getKeypoints().size()-1);

                    inlierCount++;
                }
            }

            LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " has " << inlierCount << " inliers out of " << trackedFeatures << " tracked features";
            return inlierCount;
        }



        Frame::Ptr prepNextFrame() {
            // get left and right images
            LOG(INFO) << "Preparing next frame";

            cv::cuda::GpuMat leftImage, rightImage;
            cv::cuda::GpuMat gpukeyPoints1, gpukeyPoints2, descriptors1, descriptors2;
            cv::Mat matDescriptors1, matDescriptors2;
            std::vector<cv::DMatch> matches, filteredMatches;
            std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
            // Features<cv::cuda::GpuMat>::Ptr features = std::make_shared<Features<cv::cuda::GpuMat>>(DetectorType::ORB, DescriptorType::BRIEF);

            // cv::Mat leftImage, rightImage;
            cv::Mat cvleftImage = dataHandler->getNextData(CameraSide::LEFT);
            cv::Mat cvrightImage = dataHandler->getNextData(CameraSide::RIGHT);
            Frame::Ptr leftFrame, rightFrame;
            if (cvleftImage.empty() || cvrightImage.empty()) {
                LOG(ERROR) << "No more data to process";
                return nullptr;
            }
            if (dataHandler->isCudaSet()) {
                // this->featureDetector->gpuStatus();
                if (this->status == voStatus::INIT) {
                    leftImage.upload(cvleftImage);
                    rightImage.upload(cvrightImage);
                    featureDetector->detectFeatures(leftImage, gpukeyPoints1, descriptors1);
                    featureDetector->detectFeatures(rightImage, gpukeyPoints2, descriptors2);
                    featureDetector->matchFeatures(descriptors1, descriptors2, matches);
                    featureDetector->removeOutliers(matches, filteredMatches);
                    featureDetector->convertGPUKpts(keyPoints1, gpukeyPoints1);
                    featureDetector->convertGPUKpts(keyPoints2, gpukeyPoints2);
                    descriptors1.download(matDescriptors1);
                    descriptors2.download(matDescriptors2);
                    leftFrame = std::make_shared<Frame>(Frame::createFrameID(), Sophus::SE3d(), keyPoints1, matDescriptors1, this->intrinsics, filteredMatches, cvleftImage);
                    rightFrame = std::make_shared<Frame>(Frame::createFrameID(), Sophus::SE3d(), keyPoints2, matDescriptors2, this->intrinsics, filteredMatches, cvrightImage);
                    leftFrame->rightFrame = rightFrame;
                } else {
                    std::vector<cv::KeyPoint> emptyKpts = {};
                    std::vector<cv::DMatch> emptyMatches = {};
                    leftFrame = std::make_shared<Frame>(Frame::createFrameID(), Sophus::SE3d(), emptyKpts, cv::Mat(), this->intrinsics, emptyMatches, cvleftImage);
                    rightFrame = std::make_shared<Frame>(Frame::createFrameID(), Sophus::SE3d(), emptyKpts, cv::Mat(), this->intrinsics, emptyMatches, cvrightImage);
                    leftFrame->rightFrame = rightFrame;
                }
            } else {
                LOG(ERROR) << "CPU framePrep not implemented yet";
            }
            return leftFrame;
        }

        /*
        Initialize the map with the first two frames of stereo
        
        */


        bool buildInitMap() {
            // Triangulate points based on the LR stereo images
            std::vector<cv::DMatch> matches = currFrame->getLRMatches();
            bool ret = Handler3D->triangulateAll(currFrame, currFrame->rightFrame, matches);
             currFrame->rightFrame->setPose(currFrame->getRightPoseInWorldFrame());

            if (!ret) {
                return false;
            }
            // update map with all 3d points 
            insertMPfromFrame(currFrame);
            LOG(INFO) << "Map initialized";
            return true;
        }

        void insertMPfromFrame(Frame::Ptr frame) {
            auto obsMapPoints = frame->getObsMapPoints();
            for (auto &obsMapPoint : obsMapPoints) {
                map->insertMapPoint(obsMapPoint.second);
            }
        }



        bool voLoop();


};


int main() {


    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    // convert to string
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::string now_str = std::ctime(&now_c);
    std::string logPath = "logs/" + now_str + "_VO.log";
    // google::SetLogDestination(google::WARNING,"");
    google::InitGoogleLogging("YET_ANOTHER_VO");
    std::string configFile = "../config/KITTI_stereo.json";
    VO<cv::cuda::GpuMat> vo(configFile, dataset::kitti);

    // vo.runVO();
      // Lh.viz->viewerThread = std::thread(std::bind(&Viewer::plotterLoop, Lh.viz));
    std::thread VOThread = std::thread(std::bind(&VO<cv::cuda::GpuMat>::runVO, vo));
    vo.viewer->viewerRun();
    // google::LogMessage::Flush()
    google::FlushLogFiles(google::GLOG_INFO);

    return 0;
}
