

#include "Utils.hpp"
#include "Frame.hpp"
#include "3DHandler.hpp"
#include "MapPoint.hpp"
#include "Camera.hpp"
#include "Features.hpp"
#include "Viz.hpp"
#include "Map.hpp"


enum dataset {kitti, tum, euroc};


enum voStatus {INIT, TRACKING, ERROR, RESET};


template <typename T>
class VO {
    private:

        Map::Ptr map;
        Intrinsics::Ptr intrinsics;
        DataHandler::Ptr dataHandler;
        Viewer::Ptr viewer;
        typename Features<T>::Ptr featureDetector;
        // Features<c>::Ptr featureDetector;
        _3DHandler::Ptr Handler3D;
        std::string logsPath;
        bool debugMode;
        voStatus status;
        unsigned long debugSteps;
        Frame::Ptr currFrame;
        Frame::Ptr prevFrame;

    public:
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
                        status = voStatus::TRACKING;
                    } else {
                        
                    }
                }
            }
        
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
            if (cvleftImage.empty() || cvrightImage.empty()) {
                LOG(ERROR) << "No more data to process";
                return nullptr;
            }
            if (dataHandler->isCudaSet()) {
                // this->featureDetector->gpuStatus();
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
            } else {
                
                // featureDetector->detectFeatures(cvleftImage, keyPoints1, matDescriptors1);
                // featureDetector->detectFeatures(cvrightImage, keyPoints2, matDescriptors2);
                // featureDetector->matchFeatures(matDescriptors1, matDescriptors2, matches);
                // featureDetector->removeOutliers(matches, filteredMatches);
                LOG(ERROR) << "CPU framePrep not implemented yet";
            }

            Frame::Ptr leftFrame = std::make_shared<Frame>(Frame::createFrameID(), Sophus::SE3d(), keyPoints1, matDescriptors1, this->intrinsics, filteredMatches);
            Frame::Ptr rightFrame = std::make_shared<Frame>(Frame::createFrameID(), Sophus::SE3d(), keyPoints2, matDescriptors2, this->intrinsics, filteredMatches);


            leftFrame->rightFrame = rightFrame;
            return leftFrame;
        }

        /*
        Initialize the map with the first two frames of stereo
        
        */


        bool buildInitMap() {
            // Triangulate points based on the LR stereo images
            std::vector<cv::DMatch> matches = currFrame->getLRMatches();
            Handler3D->triangulateAll(currFrame, currFrame->rightFrame, matches);

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
    vo.runVO();
    return 0;
}
