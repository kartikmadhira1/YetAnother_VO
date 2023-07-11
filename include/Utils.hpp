#ifndef UTILS_H
#define UTILS_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <ctype.h>
#include <map>
#include <mutex>
#include <chrono>
#include <jsoncpp/json/json.h>
#include <sophus/se3.hpp>
#include <Eigen/Dense>
#include <boost/filesystem.hpp>
#include <glog/logging.h>
#include "opencv4/opencv2/core.hpp"
// #include <opencv2/calib3d/calib3d.hpp>
// #include <opencv2/video/tracking.hpp>
#include "opencv4/opencv2/imgproc/imgproc.hpp"
#include "opencv4/opencv2/highgui/highgui.hpp"
#include "opencv4/opencv2/features2d.hpp"
#include "opencv4/opencv2/core.hpp"
#include "opencv4/opencv2/core/eigen.hpp"
#include <opencv4/opencv2/cudaimgproc.hpp>


// common typedefs
typedef Eigen::Matrix<double, 3, 4> Mat34;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatXX;
typedef Eigen::Matrix<double, 3, 1> Vec3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VecX;


void parseCalibString(std::string string, cv::Mat &cvMat);



// void getCalibParams(std::string _psath, Intrinsics &calib);


std::vector<boost::filesystem::path> getFilesInFolder(const std::string &path);


std::string type2str(int type);


#endif // TODOITEM_H
