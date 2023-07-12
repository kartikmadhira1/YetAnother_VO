#include "../include/Map.hpp"






// Add frame to the frame_id -> frame mapping with FramesType variable
void Map::insertKeyFrame(Frame::Ptr currFrame) {
    std::unique_lock<std::mutex> lck(mapLock);

    this->currentFrame = currFrame;
    if (frames.find(currFrame->getFrameID()) == frames.end()) {
        frames.insert(std::make_pair(currentFrame->getFrameID(), currentFrame));
    } else {
        frames.insert(std::make_pair(currentFrame->getFrameID(), currentFrame));
    }
    if (activeF.find(currFrame->getFrameID()) == activeF.end()) {
        activeF.insert(std::make_pair(currentFrame->getFrameID(), currentFrame));
    } else {
        activeF.insert(std::make_pair(currentFrame->getFrameID(), currentFrame));
    }
}

void Map::insertMapPoint(MapPoint::Ptr mp) {
    std::unique_lock<std::mutex> lck(mapLock);

    if (landmarks.find(mp->getMapPointID()) == landmarks.end()) {
        landmarks.insert(std::make_pair(mp->getMapPointID(), mp));
    } else {
        landmarks.insert(std::make_pair(mp->getMapPointID(), mp));
    }
    if (activeL.find(mp->getMapPointID()) == activeL.end()) {
        activeL.insert(std::make_pair(mp->getMapPointID(), mp));
    } else {
        activeL.insert(std::make_pair(mp->getMapPointID(), mp));
    }
}


bool Map::resetActive() {
    // empty both active frames and landmarks
    std::unique_lock<std::mutex> lck(mapLock);

    activeL.clear();
    activeF.clear();
}


Map::Ptr Map::createMap() {
    Map::Ptr _map = std::make_shared<Map>();
    return _map;
}

Map::FramesType Map::getActiveFrames() {
    std::unique_lock<std::mutex> lck(mapLock);
    return activeF;
}

Map::FramesType Map::getFrames() {
    std::unique_lock<std::mutex> lck(mapLock);
    return frames;
}

Map::LandMarksType Map::getActiveMPs() {
    std::unique_lock<std::mutex> lck(mapLock);
    return activeL;

}

Map::LandMarksType Map::getMPs() {
    std::unique_lock<std::mutex> lck(mapLock);
    return landmarks;
}