#include "../include/DataHandler.hpp"



void KITTI::loadConfig(std::string &_path) {
    std::ifstream ifs(_path);
    Json::Reader reader;
    Json::Value value;
    try {
        reader.parse(ifs, value);
    } catch (std::exception &e) {
        LOG(ERROR) << "Error parsing config file" << std::endl;
    }
    this->basePath = value["basePath"].asString();
    this->seqNo = value["sequence"].asString();
    this->calibPath = basePath  + seqNo + "/calib.txt";
    this->camType = value["cameraType"].asString();
    if (this->camType == "mono") {
        leftImagesPath = basePath + seqNo + "/image_0/";
        // rightImagesPath = basePath + seqNo + "/images_1/";
        rightImagesPath = "";
    } else {
        this-> isStereo = true;
        leftImagesPath = basePath + seqNo + "/image_0/";
        rightImagesPath = basePath + seqNo + "/image_1/";
    }
}


Intrinsics::ptr KITTI::getCalibParams() {
    Intrinsics::ptr calib = std::make_shared<Intrinsics>();
    std::vector<std::string> stringVector;
    std::string line;
    std::ifstream _file(this->calibPath);
    if (_file.good()) {
        int _i = 0;
        if (_file.is_open()) {
            for (int i=0; i<2;i++) {
                    cv::Mat cvMat =  cv::Mat(4, 4, CV_64F);
                    getline(_file, line);
                    this->parseCalibString(line, cvMat);
                    if (i == 0) {
                        Camera left(cvMat);
                        calib->Left = left;
                    } else {
                        Camera right(cvMat);
                        calib->Right = right;
                    }
                }
                
            }
    } else {
        LOG(ERROR) << "Error opening calib file:" << this->calibPath << std::endl;    }
    return calib;
}


void KITTI::parseCalibString(std::string string, cv::Mat &cvMat) {
    std::vector<double> matValues;
    std::string s;
    std::istringstream f(string);

    while(getline(f, s, ' ')) {
        if (s != " ") {
                double d;
                try {
                    d = std::stod(s);
                    matValues.emplace_back(d);

                }
                catch(std::exception &e) {
                    std::cout << e.what() <<std::endl;
                }
        }
    }

    cvMat.at<double>(0,0) = matValues[0]; cvMat.at<double>(0,1) = matValues[1]; cvMat.at<double>(0,2) = matValues[2]; cvMat.at<double>(0,3) = matValues[3]; 
    cvMat.at<double>(1,0) = matValues[4]; cvMat.at<double>(1,1) = matValues[5]; cvMat.at<double>(1,2) = matValues[6]; cvMat.at<double>(1,3) = matValues[7]; 
    cvMat.at<double>(2,0) = matValues[8]; cvMat.at<double>(2,1) = matValues[9]; cvMat.at<double>(2,2) = matValues[10]; cvMat.at<double>(2,3) = matValues[11]; 
    cvMat.at<double>(3,0) = matValues[12]; cvMat.at<double>(3,1) = matValues[13]; cvMat.at<double>(3,2) = matValues[14]; cvMat.at<double>(3,3) = matValues[15]; 


}


void KITTI::generatePathTrains() {
    std::vector<boost::filesystem::path> filesLeft;
    std::vector<boost::filesystem::path> filesRight;
    if (this->isStereo) {
        filesLeft = getFilesInFolder(leftImagesPath);
        for (auto &eachLeftPath : filesLeft) {
            leftImageTrain.push_back(eachLeftPath.string());
        }
        // Generate train for Right
        filesRight = getFilesInFolder(rightImagesPath);
        for (auto &eachRightPath : filesRight) {
            rightImageTrain.push_back(eachRightPath.string());
        }

    } else {
        filesLeft = getFilesInFolder(leftImagesPath);
        for (auto &eachLeftPath : filesLeft) {
            leftImageTrain.push_back(eachLeftPath.string());
        }
    }
}




cv::Mat KITTI::getNextData(CameraSide cam) {
    std::string path;
    if (cam == CameraSide::LEFT) {
        path = *leftImageTrainIt;
    } else {
        path = *rightImageTrainIt;
    }

    try {
        cv::Mat image = cv::imread(path);
        if (cam == CameraSide::LEFT) {
            leftImageTrainIt++;
        } else {
            rightImageTrainIt++;
        }
        return image;
    } catch (std::exception &e) {
        // log error
        LOG(ERROR) << "Error reading file: " << path << "Error:\n" << e.what() << std::endl;
        return cv::Mat();
    }
}