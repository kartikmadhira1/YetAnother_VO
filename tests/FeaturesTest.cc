#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../include/Features.hpp"
#include "../include/DataHandler.hpp"


 
TEST(DataHandlerCheck, checkInstrinsicIntegrity) {
    //get current datetime
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    // convert to string
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::string now_str = std::ctime(&now_c);
    std::string logPath = "/home/kartik/devel/projects/YA_VO_2/logs/" + now_str + "_DataHandlerTests.log";
    google::SetLogDestination(0, logPath.c_str());
    // google::SetLogDestination(google::WARNING,"");
    google::InitGoogleLogging("DataHandlerTest");
    std::string configPath = "/home/kartik/devel/projects/YA_VO_2/config/KITTI_stereo.json";
    KITTI kitti(configPath);
    kitti.generatePathTrains();
    // cv::cuda::Stream stream;

    cv::Mat leftImg = kitti.getNextData(CameraSide::LEFT);
    cv::Mat rightImg = kitti.getNextData(CameraSide::RIGHT);

    cv::cuda::GpuMat leftImgGpu, rightImgGpu,kp1, kp2 ,desc1, desc2;
    // std::vector<cv::KeyPoint> kp1, kp2;
    leftImgGpu.upload(leftImg);
    // rightImgGpu.upload(rightImg);
    // Features<cv::cuda::GpuMat> features(DetectorType::ORB, DescriptorType::BRIEF, true);
    Features<cv::cuda::GpuMat>::Ptr features = std::make_shared<Features<cv::cuda::GpuMat>>(DetectorType::ORB, DescriptorType::BRIEF, true);
    features->detectFeaturesGPU(leftImgGpu, kp1, desc1);
}

