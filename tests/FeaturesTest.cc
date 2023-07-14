#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../include/Features.hpp"
#include "../include/DataHandler.hpp"


 
TEST(FeaturesCheck, checkMatchingAlgoGPU) {
    //get current datetime
    std::string configPath = "../config/KITTI_stereo.json";

    KITTI kitti(configPath);
    kitti.generatePathTrains();
    // cv::cuda::Stream stream;
    // get time in between
    Features<cv::cuda::GpuMat>::Ptr features = std::make_shared<Features<cv::cuda::GpuMat>>(DetectorType::ORB, DescriptorType::BRIEF);

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    // for (int i=0; i<kitti.getTotalFrames(); i++) {
    cv::Mat leftImg = kitti.getNextData(CameraSide::LEFT);
    cv::Mat rightImg = kitti.getNextData(CameraSide::RIGHT);

    cv::cuda::GpuMat leftImgGpu, rightImgGpu, kp1, kp2, desc1, desc2;
    // std::vector<cv::KeyPoint> kp1, kp2;
    leftImgGpu.upload(leftImg);
    rightImgGpu.upload(rightImg);

    features->detectFeatures(leftImgGpu, kp1, desc1);
    features->detectFeatures(rightImgGpu, kp2, desc2);

    // what are the sizes of kp1 and kp2 and the corresponding cpu versions


    std::vector<cv::DMatch> matches;
    features->matchFeatures(desc1, desc2, matches);
        // get time in between
     
    // }
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    LOG(INFO) << "Time taken for feature detection and matching with GPU: " << elapsed_seconds.count() << "s\n";
    std::cout << "Time taken for feature detection and matching with GPU: " << elapsed_seconds.count() << "s\n";
  
}


TEST(FeaturesCheck, checkOutlierRemoval) {

    std::string configPath = "../config/KITTI_stereo.json";
    KITTI kitti(configPath);
    // kitti.generatePathTrains();
    // cv::cuda::Stream stream;
    // get time in between
    Features<cv::cuda::GpuMat>::Ptr features = std::make_shared<Features<cv::cuda::GpuMat>>(DetectorType::ORB, DescriptorType::BRIEF);

    cv::Mat leftImg = kitti.getNextData(CameraSide::LEFT);
    cv::Mat rightImg = kitti.getNextData(CameraSide::RIGHT);

    cv::cuda::GpuMat leftImgGpu, rightImgGpu, kp1, kp2, desc1, desc2;
    // std::vector<cv::KeyPoint> kp1, kp2;
    leftImgGpu.upload(leftImg);
    rightImgGpu.upload(rightImg);

    features->detectFeatures(leftImgGpu, kp1, desc1);
    features->detectFeatures(rightImgGpu, kp2, desc2);

    features->gpuStatus();
   

    std::vector<cv::DMatch> matches;
    features->matchFeatures(desc1, desc2, matches);
        // get time in between
    cv::Mat fullImg, inlierImg, leftImgOutlierCheck, rightImgOutlierCheck;
    features->drawMatches(leftImgGpu, rightImgGpu, kp1, kp2, matches, fullImg);

    cv::imwrite("/home/kartik/devel/projects/YA_VO_2/tests/fullImg.png", fullImg);
    std::vector<cv::DMatch> inlierMatches;
    features->removeOutliers(matches, inlierMatches);

    std::vector<cv::KeyPoint> kp1CPU, kp2CPU;
    //https://stackoverflow.com/a/10768362/6195275
    features->convertGPUKpts(kp1CPU, kp1);
    features->convertGPUKpts(kp2CPU, kp2);
    rightImgGpu.download(rightImgOutlierCheck);
    for (auto &eachMatch : inlierMatches) {
        cv::circle(rightImgOutlierCheck,kp2CPU[eachMatch.trainIdx].pt, 2, cv::Scalar(0, 255, 0), 3);
        // cv::circle(fullImg,kp2CPU[eachMatch.queryIdx].pt, 2, cv::Scalar(0, 255, 0), 2);
    }


    // features->drawMatches(leftImgGpu, rightImgGpu, kp1, kp2, inlierMatches, inlierImg);
    cv::imwrite("/home/kartik/devel/projects/YA_VO_2/tests/rightImgOutlierCheck.png", rightImgOutlierCheck);
  
}

