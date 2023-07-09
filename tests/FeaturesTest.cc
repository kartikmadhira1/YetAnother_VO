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
    // cv::KeyPoint kp1, kp2;
    cv::imwrite("/home/kartik/devel/projects/YA_VO_2/tests/leftImg.png", leftImg);
    cv::imwrite("/home/kartik/devel/projects/YA_VO_2/tests/rightImg.png", rightImg);
    cv::cuda::GpuMat leftImgGpu, rightImgGpu, kp1, kp2, desc1, desc2;
    // std::vector<cv::KeyPoint> kp1, kp2;
    leftImgGpu.upload(leftImg);
    rightImgGpu.upload(rightImg);
  
    Features<cv::cuda::GpuMat>::Ptr features = std::make_shared<Features<cv::cuda::GpuMat>>(DetectorType::ORB, DescriptorType::BRIEF, true);
    features->detectFeaturesGPU(leftImgGpu, kp1, desc1);
    features->detectFeaturesGPU(rightImgGpu, kp2, desc2);
    std::vector<cv::DMatch> matches;
    features->matchFeaturesGPU(desc1, desc2, matches);
    std::cout << kp1.size() << std::endl;
    std::cout << kp2.size() << std::endl;

    cv::Mat outImgCPU;
    features->drawMatchesGPU(leftImgGpu, rightImgGpu, kp1,  kp2, matches, outImgCPU);
    cv::imwrite("/home/kartik/devel/projects/YA_VO_2/tests/matches.png", outImgCPU);
}

