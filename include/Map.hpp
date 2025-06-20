#ifndef MAP_H
#define MAP_H

#include <unordered_map>
#include "Frame.hpp"
#include "MapPoint.hpp"


class Map {
    public:
        // 3d points stored as a map
        typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandMarksType;
        typedef std::unordered_map<unsigned long, Frame::Ptr> FramesType;
        typedef std::shared_ptr<Map> Ptr;

        Map() {}
        // initialize map with initial pose 
        void initMap();
        // Insert frames and mappoint to the whole map
        void insertKeyFrame(Frame::Ptr fr);
        void insertMapPoint(MapPoint::Ptr mp);
        FramesType getActiveFrames();
        FramesType getFrames();
        LandMarksType getActiveMPs();
        LandMarksType getMPs();
        static Map::Ptr createMap();
        bool resetActive();
    private:
        std::mutex mapLock;
        LandMarksType landmarks;
        FramesType frames;
        LandMarksType activeL;
        FramesType activeF;
        Frame::Ptr currentFrame = nullptr;

};

#endif // TODOITEM_H
