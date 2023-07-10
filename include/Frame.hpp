#include "../include/Utils.hpp"
#include "../include/DataHandler.hpp"



class Frame  {
    private:
        Sophus::SE3d pose;
        unsigned long frameID;      
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        // its a frameiD -> <source ID, dst ID > pair for keypoints
        // basically gives you access to which keypoint in src and dst match 
        std::map<unsigned long, std::pair<std::vector<int>, std::vector<int>>> matchKpMap;
    
    public:
        typedef std::shared_ptr<Frame> Ptr;
        Frame() {}
        Frame(unsigned long _frameID, Sophus::SE3d _pose, std::vector<cv::KeyPoint> _keypoints, cv::Mat _descriptors) {
            frameID = _frameID;
            pose = _pose;
            keypoints = _keypoints;
            descriptors = _descriptors;
        }
        
        Sophus::SE3d getPose() {
            return pose;
        }

        void setPose(Sophus::SE3d _pose) {
            this->pose = pose;
        }

        bool updateMatchesMap (unsigned long _frameID, std::vector<cv::DMatch> &matches) {
            std::vector<int> src;
            std::vector<int> dst;
            for (auto &match : matches) {
                src.push_back(match.queryIdx);
                dst.push_back(match.trainIdx);
            }

            if (matchKpMap.find(frameID) != matchKpMap.end()) {
                LOG(ERROR) << "Frame ID:" << this->frameID << "already has matches from frame ID: " << frameID;
                return false;
            }
            matchKpMap[frameID] = std::make_pair(src, dst);
        }
};