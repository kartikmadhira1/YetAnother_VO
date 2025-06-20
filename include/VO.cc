

#include "Utils.hpp"
#include "Frame.hpp"
#include "3DHandler.hpp"
#include "MapPoint.hpp"
#include "Camera.hpp"
#include "Features.hpp"
#include "Viz.hpp"
#include "Map.hpp"
#include "OptFlow.hpp"
#include "PnP.hpp"

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

        bool relVelSet = false;
        // Thresholds
        int minInlierCount = 30;

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
                  
                        prevFrame = currFrame;
                        status = voStatus::TRACKING;
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
                currFrame->setPose(relativeMotion*prevFrame->getPose());


                // // If relative velocity is not set, we need to find matches in the second frame and then set 
                // if (!relVelSet) {
                //     LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " - Initializing SECOND frame";
                //     initSecFrame();
                //     relVelSet = true;
                // } else {
                //     int optFlowInlierCount = track();
                // }
                int optFlowInlierCount = track();
                // 5. If more than 20% points are inliers, register frame as regular frame, update the mapPoint with new observation and update frame with observation

                // optimize the pose of the current frame using PnP provided there are enough inliers
                int optimizedInliers = optimizeCurrPose();

                if (optimizedInliers < minInlierCount) {
                    LOG(ERROR) << "Frame ID: " << currFrame->getFrameID() << " - TRACKING FAILED";
                    LOG(INFO) << "Triangulating new points";

                    // before new features, capture index of last frame keypoints +1
                    int indexLastFrameKp = currFrame->getKeypoints().size();

                    // new features detect+match with right frame with masking of already detected features
                    detectNewFeatures(true, currFrame);

                    // triangulate new points
                    triangulateNewPoints(indexLastFrameKp);

                }

                // register as a keyframe
                map->insertKeyFrame(currFrame);
                map->insertKeyFrame(currFrame->rightFrame);
                viewer->addCurrentFrame(currFrame);
                relativeMotion = currFrame->getPose()*prevFrame->getPose().inverse();
           
                prevFrame = currFrame;
           
                //6. If inliers not enough, mask the points that are now features and detect new features that are masked with the already detected features

            }
        
            viewer->updateMap();

        }

    

        bool detectNewFeatures(bool trackedFrame, Frame::Ptr frame) {
            cv::Mat mask(frame->getRawImg().size(), CV_8UC1, 255);
            cv::Mat imgCopy = frame->getRawImg();
            cv::cvtColor(imgCopy, imgCopy, cv::COLOR_GRAY2BGR);
            cv::Mat imgCopy2 = frame->getRawImg();
            cv::cvtColor(imgCopy2, imgCopy2, cv::COLOR_GRAY2BGR);
            // if this is a tracked frame, mask the current features
            if (trackedFrame) {
                auto currFrameKpts = frame->getKeypoints();
                // create a mask
                for (auto &currFrameKpt : currFrameKpts) {
                    cv::rectangle(mask, currFrameKpt.pt - cv::Point2f(10, 10), currFrameKpt.pt + cv::Point2f(10, 10), 0, cv::FILLED);
                    cv::circle(imgCopy, currFrameKpt.pt, 3, cv::Scalar(0, 0, 255), 2);
                }
            }

            // download mask to gpu
            cv::cuda::GpuMat maskGPU;
            maskGPU.upload(mask);

            cv::cuda::GpuMat leftImage, rightImage;
            cv::cuda::GpuMat gpukeyPoints1, gpukeyPoints2, descriptors1, descriptors2;
            cv::Mat matDescriptors1, matDescriptors2;
            std::vector<cv::DMatch> matches, filteredMatches;
            std::vector<cv::KeyPoint> keyPoints1, keyPoints2;

            cv::Mat cvLeftImage = frame->getRawImg();
            cv::Mat cvRightImage = frame->rightFrame->getRawImg();

            leftImage.upload(cvLeftImage);
            rightImage.upload(cvRightImage);
            featureDetector->detectFeatures(leftImage, gpukeyPoints1, descriptors1, maskGPU);
            featureDetector->detectFeatures(rightImage, gpukeyPoints2, descriptors2);
            featureDetector->matchFeatures(descriptors1, descriptors2, matches);
            featureDetector->removeOutliers(matches, filteredMatches, 3);
            featureDetector->convertGPUKpts(keyPoints1, gpukeyPoints1);
            featureDetector->convertGPUKpts(keyPoints2, gpukeyPoints2);

            // set new feautres to the frame
            int indexLastFrameKp = frame->getKeypoints().size();
            int indexLastFrameCopy = indexLastFrameKp;

            for (auto &keyPoint : keyPoints1) {
                frame->addKeypoint(keyPoint);
                frame->setFeatureInlierFlag(indexLastFrameKp, true);
                cv::circle(imgCopy, keyPoint.pt, 5, cv::Scalar(255, 0, 0), 2);

                indexLastFrameKp++;
            }

            for (auto &keyPoint : keyPoints2) {
                frame->rightFrame->addKeypoint(keyPoint);
                // frame->rightFrame->setFeatureInlierFlag(indexLastFrameKp, true);
                // indexLastFrameKp++;
            }


            std:vector<cv::DMatch> filteredMatchesCheck;

            for (auto &eachMatch : filteredMatches) {
                cv::DMatch newMatch;
                newMatch.queryIdx = indexLastFrameCopy + eachMatch.queryIdx;
                newMatch.trainIdx = eachMatch.trainIdx;
                newMatch.distance = eachMatch.distance;
                filteredMatchesCheck.push_back(newMatch);
            }

            auto lFeat = frame->getKeypoints();
            auto rFeat = frame->rightFrame->getKeypoints();

            featureDetector->drawMatches(cvLeftImage, cvRightImage, lFeat, rFeat, filteredMatchesCheck, imgCopy2);
            cv::imwrite("image" + std::to_string(frame->getFrameID())+ "_lrCheck.png", imgCopy2);


            frame->rightFrame->setAllInliers(true);
            // set LR matches to filtered matches
            frame->setLRmatches(filteredMatches);
            if (this->debugMode) {
                cv::imwrite("image" + std::to_string(frame->getFrameID())+ "_mask_check.png", imgCopy);
            }
            return true;
        } 

        bool triangulateNewPoints(int indexLastFrameKp) {
            // Triangulate points based on the LR stereo images
            std::vector<cv::DMatch> matches = currFrame->getLRMatches();
            bool ret = Handler3D->triangulateAll(currFrame, currFrame->rightFrame, matches, true, indexLastFrameKp);
            currFrame->rightFrame->setPose(currFrame->getRightPoseInWorldFrame());

            if (!ret) {
                return false;
            }
            // update map with all 3d points 
            insertMPfromFrame(currFrame);
            LOG(INFO) << "New points triangulated: " << currFrame->rightFrame->getObsMapPoints().size();
            return true;
        }


        int optimizeCurrPose() {
            // set up g2o

            typedef g2o::BlockSolver_6_3 BlockSolverType;
            typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
                LinearSolverType;
            auto solver = new g2o::OptimizationAlgorithmLevenberg(
                std::make_unique<BlockSolverType>(
                    std::make_unique<LinearSolverType>()));
            g2o::SparseOptimizer optimizer;
            optimizer.setAlgorithm(solver);

            // there is only one vertex => current frame pose
            PnPVertex* v = new PnPVertex();
            v->setId(0);
            v->setEstimate(currFrame->getPose());
            optimizer.addVertex(v);

            // add edges
            int index = 1;
            std::vector<PnPEdgeProjection*> edges;

            // For each feature in current frame => get corresponding 3d point
            auto currFrameKpts = currFrame->getKeypoints();
            auto currFrameObs = currFrame->getObsMapPoints();
            // get kp size
            int edgeID = 0;
            std::map<int, int> edgeToKpID;
            for (int i=0; i< currFrameKpts.size();i++) {
                // get current frame keypoint
                auto currFrameKpt = currFrameKpts[i];
                // get corresponding 3d point
                // will have to iterate through all observations and find which mapPoint has this kpID
                int mpID = currFrame->getMpIDfromKpID(i);
                if (mpID == -1) {
                    LOG(ERROR) << "Frame ID: " << currFrame->getFrameID() << " does not have a 3d point corresponding to keypoint ID: " << i;
                    continue;
                }
                auto mapPoint = currFrameObs[mpID];
                // 3d point is,
                Vec3 mapPoint3D = mapPoint->getPosition();
                // 2d point in Vec2
                Vec2 framePoint2D(currFrameKpt.pt.x, currFrameKpt.pt.y);
                // create edge
                Eigen::Matrix3d K;
                intrinsics->Left.getKEigen(K);
                PnPEdgeProjection* e = new PnPEdgeProjection(mapPoint3D, K);
                e->setId(index);
                e->setVertex(0, v);
                e->setMeasurement(framePoint2D);
                e->setInformation(Eigen::Matrix2d::Identity());
                e->setRobustKernel(new g2o::RobustKernelHuber());
                edges.push_back(e);
                optimizer.addEdge(e);

                // need to map the kpID to edgeID
                edgeToKpID[edgeID] = i;
                index++;
                edgeID++;

            }
          
            // // optimize with n iterations
            const double chi2Thresh = 5.991;
            int nIterations = 4;
            int outlierCount = 0;


            for (int iteration=0; iteration < nIterations; ++iteration) {
                // optimizer.setVerbose(true);
                v->setEstimate(currFrame->getPose());

                optimizer.initializeOptimization();
                optimizer.optimize(10);
                outlierCount = 0;
                for (size_t i=0; i<edges.size(); ++i) {
                    auto e = edges[i];
                    int kpID = edgeToKpID[i];
                    if (currFrame->getFeatureInlierFlag(kpID) == false) {
                        e->computeError();
                    }
                    if (e->chi2() > chi2Thresh) {
                        currFrame->setFeatureInlierFlag(kpID, false);
                        e->setLevel(1);
                        outlierCount++;
                    } else {
                        currFrame->setFeatureInlierFlag(kpID, true);
                        e->setLevel(0);
                    };
                    if (iteration == 2) {
                        e->setRobustKernel(nullptr);
                    }
                }
            }

            LOG(INFO) << "OPTIMIZER INLIERS: Frame ID: " << currFrame->getFrameID() << " has " << outlierCount << " outliers AND " << currFrame->getKeypoints().size()- outlierCount << " inliers" ;
            
            // update the pose of the current frame
            Sophus::SE3d newPose = v->estimate();
            currFrame->setPose(newPose);

            // for all the outlier points, set the 3d point map to -1
            // for (int i=0; i<currFrame->getKeypoints().size(); i++) {
            //     if (currFrame->getFeatureInlierFlag(i) == false) {
            //         currFrame->setMpIDforKpID(i, -1);
            //     }
            // }

            return currFrame->getKeypoints().size() - outlierCount;
        }


        int track() {
            // reproject all the 3d points seen in previous frame to the current frame

            auto obsMapPoints = prevFrame->getObsMapPoints();
            std::vector<cv::Point2f> prevPts, currPts;
            std::vector<int> mapPointIndex;
            int trackedFeatures = 0;
            int notInFramePoints = 0;
            int previousPoints = 0;
            auto prevKpts = prevFrame->getKeypoints();
            // std::vector<int> prevKpIDs;
            int prevKpID = 0;
            std::map<int, int> prevPointMap;
            for (int i=0; i<prevKpts.size();i++) {
                
                // if this point has a 3d point
                if (prevFrame->getFeatureInlierFlag(i)) {
                    if (prevFrame->getMpIDfromKpID(i) != -1) {
                        // get the 3d point
                        auto mapPoint = obsMapPoints[prevFrame->getMpIDfromKpID(i)];
                        // get the 2d point
                        cv::Point2f singlePrevPt =  prevKpts[i].pt;
                        // now get 3d location of this mapPoint and reproject it to the current frame
                        Vec3 mapPoint3D = mapPoint->getPosition();
                        cv::Point2f currPt = currFrame->world2pixel(mapPoint3D, intrinsics);
                        // check if the point is within the image

                        if (currPt.x < 0 || currPt.y < 0 || currPt.x > currFrame->getRawImg().cols || currPt.y > currFrame->getRawImg().rows) {
                            notInFramePoints++;
                            continue;
                        }

                        prevPts.push_back(singlePrevPt);
                        currPts.push_back(currPt);
                        // mapPointIndex.push_back(mapPoint->getID());
                        trackedFeatures++;
                        prevPointMap[prevKpID] = i;
                        prevKpID++;
                    } else {
                        // prev and curr will have same points

                        cv::Point2f singlePrevPt =  prevKpts[i].pt;
                        prevPts.push_back(singlePrevPt);
                        currPts.push_back(singlePrevPt);
                        previousPoints++;
                        prevPointMap[prevKpID] = i;
                        prevKpID++;
                    }
                }
            }


            // log the number of tracked features for this frame pair
            LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " has: " << trackedFeatures << " reprojected 3d points from:" << obsMapPoints.size() << " 3d points ";
            LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " has: " << previousPoints << "previous points that dont have 3d points";
            LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " has: " << notInFramePoints << " rejected points that dont fit the frame ";
            // log total points
            LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " has: " << prevKpts.size() << " total points";

            int inlierCount = 0;

            // this will have to be a different function
            std::vector<uchar> flowStatus;
            cv::Mat error;
            cv::calcOpticalFlowPyrLK(
                prevFrame->getRawImg(), currFrame->getRawImg(), prevPts,
                currPts, flowStatus, error, cv::Size(11, 11), 3,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                                0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);

            // first empty the keypoints          
            currFrame->clearKeypoints();
            // Need to make sure flowstatus size and prevKpts size are same
            
            for (int i=0; i<flowStatus.size();i++) {
                if(flowStatus[i]) {
                    // Flow is good for this point, add this point as feature to the current frame


                    // check if this mappoint exists, if not then create new one
                    // get original keypoint id
                    int originalKpID = prevPointMap[i];
                    if (prevFrame->getMpIDfromKpID(originalKpID) !=-1) {
                        currFrame->addKeypoint(cv::KeyPoint(currPts[i], 3));

                        obsMapPoints[prevFrame->getMpIDfromKpID(originalKpID)]->addObservation(currFrame->getFrameID(), inlierCount);
                        // add this point as observation to the mapPoint
                        currFrame->addObservation(obsMapPoints[prevFrame->getMpIDfromKpID(originalKpID)]);
                                            inlierCount++;

                    } 
            

                }
            }

            currFrame->setAllInliers(true);

            LOG(INFO) << "TRACKING INLIERS: Frame ID: " << currFrame->getFrameID() << " has " << inlierCount << " inliers out of " << trackedFeatures << " tracked features";
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
                    featureDetector->removeOutliers(matches, filteredMatches, 5);
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

            // currFrame->setAllInliers(false);
            // currFrame->rightFrame->setAllInliers(false);
            // // add keypoints only that are matches
            // for (auto &eachMatch : matches) {
             
            //     // set the feature inlier flag to true
            //     currFrame->setFeatureInlierFlag(eachMatch.queryIdx, true);
            //     // do same for right frame
            //     currFrame->rightFrame->setFeatureInlierFlag(eachMatch.trainIdx, true);
            // }



            bool ret = Handler3D->triangulateAll(currFrame, currFrame->rightFrame, matches,false, 0);
            // currFrame->rightFrame->setPose(currFrame->getRightPoseInWorldFrame());

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
