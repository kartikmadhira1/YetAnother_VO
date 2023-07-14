#include <gtest/gtest.h>
#include <glog/logging.h>
#include "../include/DataHandler.hpp"
#include "../include/Frame.hpp"

 



TEST(FramesCheck, checkPoses) {
    std::string configPath = "../config/KITTI_stereo.json";
    KITTI kitti(configPath);


    // for (int i=0; i<20; i++) {
    std::string leftName = kitti.getCurrImagePath(CameraSide::LEFT);
    std::string rightName = kitti.getCurrImagePath(CameraSide::RIGHT);
    EXPECT_NE(leftName, rightName);
    EXPECT_TRUE(kitti.assertFilename(leftName, rightName));
    kitti.getNextData(CameraSide::LEFT);
    kitti.getNextData(CameraSide::RIGHT);
    // unsigned long frameId = Frame::s
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches;
    Frame::Ptr frame = std::make_shared<Frame>(Frame::createFrameID(), Sophus::SE3d(), keypoints1, cv::Mat(), kitti.getCalibParams(), matches, cv::Mat());
    Frame::Ptr rightFrame = std::make_shared<Frame>(Frame::createFrameID(), Sophus::SE3d(), keypoints2, cv::Mat(), kitti.getCalibParams(), matches, cv::Mat());
    frame->rightFrame = rightFrame;
    std::cout << frame->getRightPoseInWorldFrame().matrix() << std::endl;
    // std::cout << Sophus::SE3d().matrix() << std::endl;

    // }   
}