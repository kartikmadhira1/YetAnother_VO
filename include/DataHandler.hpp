#ifndef DATA_HANDLER_H
#define DATA_HANDLER_H

#include <string>
#include "../include/Utils.hpp"

/*
Abstract Interface for Data handling different datasets
*/

enum CameraSide {
    LEFT,
    RIGHT
}


class DataHandler {
    public:
        virtual cv::Mat getNextData(CameraSide cam) = 0;
        virtual Instrinsics getCalibParams() = 0;
        virtual void loadConfig(std::string &path) = 0;
};