#ifndef MAPPOINT_H
#define MAPPOINT_H


#include "Utils.hpp"




class MapPoint {
    private:
        unsigned long mapPointID;
        // How many frames have observed this map point
        unsigned long obsCount;
        Vec3 position;
        // what Frame ID and what keypoint ID in that frame ID
        std::map<unsigned long, int> frameIDToKpID;
        std::mutex mapPointMutex;

    public:
        typedef std::shared_ptr<MapPoint> Ptr;
        MapPoint() {}
        MapPoint(unsigned long _mapPointID, cv::Point3d _position) {
            mapPointID = _mapPointID;
            position[0] = _position.x;
            position[1] = _position.y;
            position[2] = _position.z;            
            obsCount = 0;
        }
        MapPoint(unsigned long _mapPointID, Vec3 _position) {
            mapPointID = _mapPointID;
            position = _position;
            obsCount = 0;
        }
        void addObservation(unsigned long frameID, int kpID) {
            std::unique_lock<std::mutex> lock(mapPointMutex);
            if (frameIDToKpID.find(frameID) != frameIDToKpID.end()) {
                LOG(ERROR) << "Frame ID: " << frameID << " already has a keypoint ID: " << frameIDToKpID[frameID];
                return;
            }
            frameIDToKpID[frameID] = kpID;
            obsCount++;
        }

        int getKpID(unsigned long frameID) {
            std::unique_lock<std::mutex> lock(mapPointMutex);
            if (frameIDToKpID.find(frameID) == frameIDToKpID.end()) {
                LOG(ERROR) << "Frame ID: " << frameID << " does not have a keypoint ID";
                return -1;
            }
            return frameIDToKpID[frameID];
        }

        static unsigned long createMapPointID() {
            static unsigned long mapPointID = 0;
            return mapPointID++;
        }

        unsigned long getMapPointID() {
            std::unique_lock<std::mutex> lock(mapPointMutex);

            return mapPointID;
        }

        unsigned long getObsCount() {
            std::unique_lock<std::mutex> lock(mapPointMutex);
            return obsCount;
        }

        Vec3 getPosition() {
            std::unique_lock<std::mutex> lock(mapPointMutex);
            return position;
        }

        void setPosition(const Vec3 _position) {
            std::unique_lock<std::mutex> lock(mapPointMutex);
            position = _position;
        }

        void setPosition(const cv::Point3d _position) {
            std::unique_lock<std::mutex> lock(mapPointMutex);
            position[0] = _position.x;
            position[1] = _position.y;
            position[2] = _position.z;
        }
};

#endif // TODOITEM_H
