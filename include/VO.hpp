#ifndef VO_H
#define VO_H


#include "Utils.hpp"
#include "Frame.hpp"
#include "3DHandler.hpp"
#include "MapPoint.hpp"
#include "Camera.hpp"
#include "Features.hpp"
#include "Viz.hpp"
#include "Map.hpp"


enum dataset {KITTI, TUM, EUROC};


enum voStatus {INIT, TRACKING, ERROR, RESET};



class VO {
    private:

        Map::Ptr map;
        Intrinsics::Ptr intrinsics;
        DataHandler dataHandler;
        Viewer::Ptr viewer;
        Features::Ptr featureDetector;
        _3DHandler::Ptr _3DHandler;

    public:
        VO(const std::string &configFile, dataset datasetType) {
            if (datasetType == KITTI) {
                LOG(INFO) << "KITTI dataset initialized";
                dataHandler = std::make_shared<KITTI>(configFile);
            } else if (datasetType == TUM) {
                LOG(ERROR) << "TUM dataset package not implemented yet";
            } else if (datasetType == EUROC) {
                LOG(ERROR) << "EUROC dataset package not implemented yet";
            } else {
                LOG(ERROR) << "Invalid dataset type";
            }
            void initModules();
        }
        void initModules();
        bool runVO();
        bool takeVOStep();
        bool buildInitMap();
        bool voLoop();

}