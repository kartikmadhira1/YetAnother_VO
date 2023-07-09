#ifndef DATA_HANDLER_H
#define DATA_HANDLER_H

#include <string>
#include "Utils.hpp"
#include "Camera.hpp"

/*
Abstract Interface for Data handling different datasets
*/

enum CameraSide {
    LEFT,
    RIGHT
};


class DataHandler {
    public:
        virtual cv::Mat getNextData(CameraSide cam) = 0;
        virtual Intrinsics::ptr getCalibParams() = 0;
        virtual void loadConfig(std::string &path) = 0;
};


/*
KITTI Dataset Handler.
*/


class KITTI : public DataHandler {

    private:
        int currFrameId;
        std::string basePath;
        std::string seqNo;
        std::string calibPath;
        std::string leftImagesPath;
        std::string rightImagesPath;
        std::vector<std::string> leftImageTrain;
        std::vector<std::string> rightImageTrain;
        std::vector<std::string>::iterator leftImageTrainIt;
        std::vector<std::string>::iterator rightImageTrainIt;
        std::string camType;
        bool isStereo;
        void parseCalibString(std::string string, cv::Mat &cvMat);


    public:
        KITTI(std::string &_configPath) {
            currFrameId = 0;
            loadConfig(_configPath);
        }
        void generatePathTrains();

        void loadConfig(std::string &_path);
        Intrinsics::ptr getCalibParams();
        cv::Mat getNextData(CameraSide cam);
        std::string getCurrImagePath(CameraSide cam);
        int getTotalFrames() {
            assert(leftImageTrain.size() == rightImageTrain.size());
            return leftImageTrain.size();
        }
        bool assertFilename(std::string leftPath, std::string rightPath);


};


#endif // TODOITEM_H
