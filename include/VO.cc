

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
        int minInlierCount = 70;

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


                // If relative velocity is not set, we need to find matches in the second frame and then set 
                if (!relVelSet) {
                    LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " - Initializing SECOND frame";
                    initSecFrame();

                    relVelSet = true;
                } else {
                    int optFlowInlierCount = track();
                }

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
                    map->insertKeyFrame(currFrame);
                    map->insertKeyFrame(currFrame->rightFrame);
                    viewer->addCurrentFrame(currFrame);
                }

                // register as a keyframe
         
                relativeMotion = currFrame->getPose()*prevFrame->getPose().inverse();
           
                prevFrame = currFrame;
           
                //6. If inliers not enough, mask the points that are now features and detect new features that are masked with the already detected features

            }
        
            viewer->updateMap();

        }

        bool initSecFrame() {
            // find features in this second frame and then match with the first ref frame
            cv::cuda::GpuMat leftImage, rightImage;
            cv::cuda::GpuMat gpukeyPoints1, gpukeyPoints2, gpukeyPointsCheck, descriptors1, descriptors2;
            cv::Mat matDescriptors1, matDescriptors2;
            std::vector<cv::DMatch> matches, filteredMatches;
            std::vector<cv::KeyPoint> keyPoints1, keyPoints2;
            cv::Mat mask(currFrame->getRawImg().size(), CV_8UC1, 255);
            cv::cuda::GpuMat maskGPU;
            maskGPU.upload(mask);
            cv::Mat cvLeftImage = prevFrame->getRawImg();
            cv::Mat cvRightImage = currFrame->getRawImg();
            
            leftImage.upload(cvLeftImage);
            rightImage.upload(cvRightImage);

            // gpukeypoints will need to have a mat of numFeaturesx6 size
            // 6 columns gpu will be x,y,1, size, angle, response
            auto lefKpts = prevFrame->getKeypoints();
            cv::Mat keypointsMatLeft(lefKpts.size(), 6, CV_32FC1);

            for (size_t i = 0; i < lefKpts.size(); i++) {
                keypointsMatLeft.at<float>(i, 0) = lefKpts[i].pt.x;
                keypointsMatLeft.at<float>(i, 1) = lefKpts[i].pt.y;
                keypointsMatLeft.at<float>(i, 2) = lefKpts[i].size;
                keypointsMatLeft.at<float>(i, 3) = lefKpts[i].angle;
                keypointsMatLeft.at<float>(i, 4) = lefKpts[i].response;
                keypointsMatLeft.at<float>(i, 5) = lefKpts[i].octave;
            }

            gpukeyPoints1.upload(keypointsMatLeft);
            
            cv::Mat lImgDesc = prevFrame->getDescriptors();
            descriptors1.upload(lImgDesc);
            // featureDetector->detectFeatures(leftImage, gpukeyPoints1, descriptors1, maskGPU ,true);
            featureDetector->detectFeatures(rightImage, gpukeyPoints2, descriptors2);
 
            featureDetector->matchFeatures(descriptors1, descriptors2, matches);

            featureDetector->removeOutliers(matches, filteredMatches, 5);

            featureDetector->convertGPUKpts(keyPoints2, gpukeyPoints2);
       
            // log filtered matches info
            LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " has " << filteredMatches.size() << " matches with reference frame";

            // now apply current frame with all the filtered matches
            currFrame->clearKeypoints();
            auto obsMapPoints = prevFrame->getObsMapPoints();
            std::vector<cv::DMatch> filteredMatchesCheck;
            int index=0;
            for (auto &eachMatch : filteredMatches) {
                // if prevPoint belongs to a matched point
                if (prevFrame->getFeatureInlierFlag(eachMatch.queryIdx)) {
                    // if such mapPoint exists with this keypoint
                    if (prevFrame->getMpIDfromKpID(eachMatch.queryIdx) != -1) {
                        // get the keypoint from the previous frame
                        // auto currKpt = curr->getKeypoints()[eachMatch.queryIdx];
                        // add the keypoint to the current frame
                        currFrame->addKeypoint(keyPoints2[eachMatch.trainIdx]);
                        cv::DMatch newMatch;
                        newMatch.queryIdx = eachMatch.queryIdx;
                        newMatch.trainIdx = index;
                        newMatch.distance = eachMatch.distance;
                        filteredMatchesCheck.push_back(newMatch);
                        // add observation to the map point
                        auto mapPoint = obsMapPoints[prevFrame->getMpIDfromKpID(eachMatch.queryIdx)];
                        mapPoint->addObservation(currFrame->getFrameID(), index);
                        currFrame->addObservation(mapPoint);

                        // obsMapPoints[prevFrame->getMpIDfromKpID(eachMatch.queryIdx)]->addObservation(currFrame->getFrameID(), index);
                        index++;

                    }
                }

            }
            cv::Mat out;
            auto lVec = prevFrame->getKeypoints();
            auto rVec = currFrame->getKeypoints();
            // temp function to see matches between previous and current frame
            featureDetector->drawMatches(cvLeftImage, cvRightImage, lVec, rVec, filteredMatchesCheck, out);
            cv::imwrite("image" + std::to_string(currFrame->getFrameID())+ "_initSecFrame_check.png", out);

            // set all inliers to true
            currFrame->setAllInliers(true);

            LOG(INFO) << "Frame ID: " << currFrame->getFrameID() << " has " << index << " observations from the previous frame";
            return true;

        }


        bool detectNewFeatures(bool trackedFrame, Frame::Ptr frame) {
            cv::Mat mask(frame->getRawImg().size(), CV_8UC1, 255);
            cv::Mat imgCopy = frame->getRawImg();
            cv::cvtColor(imgCopy, imgCopy, cv::COLOR_GRAY2BGR);
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

            // set LR matches to filtered matches
            frame->setLRmatches(filteredMatches);
            cv::imwrite("image" + std::to_string(frame->getFrameID())+ "_mask_check.png", imgCopy);
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
            LOG(INFO) << "New points triangulated";
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
                index++;

            }
          
            // // optimize with n iterations
            const double chi2Thresh = 5.991;
            int nIterations = 4;
            int outlierCount = 0;

            v->setEstimate(currFrame->getPose());

            for (int iteration=0; iteration < nIterations; ++iteration) {
                optimizer.initializeOptimization();
                // optimizer.setVerbose(true);

                optimizer.optimize(10);
                outlierCount = 0;
                for (size_t i=0; i<edges.size(); ++i) {
                    auto e = edges[i];

                    if (currFrame->getFeatureInlierFlag(i) == false) {
                        e->computeError();
                    }
                    if (e->chi2() > chi2Thresh) {
                        currFrame->setFeatureInlierFlag(i, false);
                        e->setLevel(1);
                        outlierCount++;
                    } else {
                        currFrame->setFeatureInlierFlag(i, true);
                        e->setLevel(0);
                    }
                    if (iteration == 2) {
                        e->setRobustKernel(nullptr);
                    }
                }
                LOG(INFO) << "Iteration: " << iteration << " has " << outlierCount << " outliers";
            }

            LOG(INFO) << "OPTIMIZER INLIERS: Frame ID: " << currFrame->getFrameID() << " has " << outlierCount << " outliers AND " << currFrame->getKeypoints().size()- outlierCount << " inliers" ;
            
            // update the pose of the current frame
            currFrame->setPose(v->estimate());

            // return the number of inliers
            return currFrame->getKeypoints().size() - outlierCount;
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
                                
            // first empty the keypoints          
            currFrame->clearKeypoints();
            // currFrame->clearInlierFlags();
            
            for (int i=0; i<flowStatus.size();i++) {
                if(flowStatus.at(i)==1) {
                    // Flow is good for this point, add this point as feature to the current frame
                
                    currFrame->addKeypoint(cv::KeyPoint(currPts[i], 3));
                    obsMapPoints[mapPointIndex[i]]->addObservation(currFrame->getFrameID(), inlierCount);

                    currFrame->addObservation(obsMapPoints[mapPointIndex[i]]);
                    // add this point as observation to the mapPoint

                    inlierCount++;
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

            currFrame->setAllInliers(false);
            currFrame->rightFrame->setAllInliers(false);
            // add keypoints only that are matches
            for (auto &eachMatch : matches) {
             
                // set the feature inlier flag to true
                currFrame->setFeatureInlierFlag(eachMatch.queryIdx, true);
                // do same for right frame
                currFrame->rightFrame->setFeatureInlierFlag(eachMatch.trainIdx, true);
            }



            bool ret = Handler3D->triangulateAll(currFrame, currFrame->rightFrame, matches,false, 0);
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
