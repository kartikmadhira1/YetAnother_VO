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
        virtual Intrinsics::Ptr getCalibParams() = 0;
        virtual void loadConfig(std::string &path) = 0;
        virtual std::string getBasePath() = 0;
        virtual bool getDebugMode() = 0;
        virtual unsigned long getDebugSteps() = 0;
        virtual bool isCudaSet() = 0;
        virtual int getTotalFrames() = 0;
        typedef std::shared_ptr<DataHandler> Ptr;
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
        bool cudaSet;
        bool isStereo;
        bool debugMode;
        unsigned long debugSteps;
        void parseCalibString(std::string string, cv::Mat &cvMat);


    public:
        KITTI(std::string &_configPath) {
            currFrameId = 0;
            loadConfig(_configPath);
            generatePathTrains();
        }
        void generatePathTrains();
        std::string getBasePath() {
            return basePath;
        }
        void loadConfig(std::string &_path);
        Intrinsics::Ptr getCalibParams();
        cv::Mat getNextData(CameraSide cam);
        std::string getCurrImagePath(CameraSide cam);
        int getTotalFrames() {
            assert(leftImageTrain.size() == rightImageTrain.size());
            return leftImageTrain.size();
        }
        bool assertFilename(std::string leftPath, std::string rightPath);
        bool getDebugMode() {
            return debugMode;
        }

        unsigned long getDebugSteps() {
            return debugSteps;
        }

        bool isCudaSet() {
            return cudaSet;
        }

};


#endif // TODOITEM_H
