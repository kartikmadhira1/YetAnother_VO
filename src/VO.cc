#include "VO.hpp"


void VO::initModules() {
    // Initialize all modules
    // First, check for gpu support on the system
    // If gpu support is available, use templates to initialize the Features and OpticalFlow modules
    // Else, initialize the modules with CPU

    if (checkCUDAsupport()) {
        LOG(INFO) << "CUDA support available";
        // Initialize the Features module with CUDA
        this->featureDetector = std::make_shared<Features<cv::cuda::GpuMat>>(DetectorType::ORB, DescriptorType::BRIEF);
    } else {
        LOG(INFO) << "CUDA support not available";
        // Initialize the Features module with CPU
        this->featureDetector = std::make_shared<Features<CPU>>(DetectorType::ORB, DescriptorType::BRIEF);
    }

    intrinsics = std::make_shared<Intrinsics>(dataHandler->getCalibParams());
    map = std::make_shared<Map>();
    _3DHandler = std::make_shared<_3DHandler>(intrinsics);
}





int main() {


    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    // convert to string
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::string now_str = std::ctime(&now_c);
    std::string logPath = "logs/" + now_str + "_VO.log";
    google::SetLogDestination(0, logPath.c_str());
    // google::SetLogDestination(google::WARNING,"");
    google::InitGoogleLogging("YET_ANOTHER_VO");
    VO vo("config.yaml", dataset::KITTI);

    return 0;
}