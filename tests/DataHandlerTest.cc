#include <gtest/gtest.h>
#include <glog/logging.h>
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

    Intrinsics::Ptr intrinsics = std::make_shared<Intrinsics>();
    intrinsics = kitti.getCalibParams();
        
    intrinsics->Right.printP();
    // intrinsics.Right.printK();
    std::cout << intrinsics->Right.getBaseline() << std::endl;
    EXPECT_NEAR(intrinsics->Left.K.at<double>(0,0), 718.856, 1);
    EXPECT_NEAR(intrinsics->Right.K.at<double>(0,0), 718.856, 1);
    EXPECT_NEAR(intrinsics->Left.K.at<double>(1,2), 185.216, 1);
    EXPECT_NEAR(intrinsics->Right.K.at<double>(1,2), 185.216, 1);
    EXPECT_NEAR(intrinsics->Left.getBaseline(), 0, 1);
    EXPECT_NEAR(intrinsics->Right.getBaseline(), 0, 1);

}



TEST(DataHandlerCheck, checkFilename) {
    std::string configPath = "/home/kartik/devel/projects/YA_VO_2/config/KITTI_stereo.json";
    KITTI kitti(configPath);
    kitti.generatePathTrains();
    std::cout << kitti.getTotalFrames() << std::endl;
    for (int i=0; i<20; i++) {
        std::string leftName = kitti.getCurrImagePath(CameraSide::LEFT);
        std::string rightName = kitti.getCurrImagePath(CameraSide::RIGHT);
        EXPECT_NE(leftName, rightName);
        EXPECT_TRUE(kitti.assertFilename(leftName, rightName));
        kitti.getNextData(CameraSide::LEFT);
        kitti.getNextData(CameraSide::RIGHT);
        // std::cout << leftName << std::endl;
        // std::cout << rightName << std::endl;
    }   
}