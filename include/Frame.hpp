#ifndef FRAME_HPP
#define FRAME_HPP



#include "../include/Utils.hpp"
#include "../include/DataHandler.hpp"
#include "../include/MapPoint.hpp"


class Frame  {
    private:
        Sophus::SE3d pose;
        unsigned long frameID;      
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        // its a frameiD -> <source ID, dst ID > pair for keypoints
        // basically gives you access to which keypoint in src and dst match 
        std::map<unsigned long, std::pair<std::vector<int>, std::vector<int>>> matchKpMap;
        // mappoint ID -> mappoint
        std::map<unsigned long, MapPoint::Ptr> obsMapPoints;
        cv::Mat rawImg;
        Intrinsics::Ptr intrinsics;
        std::vector<cv::DMatch> LRmatches;
        std::vector<bool> featureInlierFlag;
        // std::vector<cv::DMatch> LRmatches;
        std::mutex poseMutex; //lock whenever accesing/writing to the object.

    public:
        typedef std::shared_ptr<Frame> Ptr;
        Frame::Ptr rightFrame;

        Frame() {}


        Frame(unsigned long _frameID, Sophus::SE3d _pose, std::vector<cv::KeyPoint> _keypoints, cv::Mat _descriptors, Intrinsics::Ptr _intrinsics, std::vector<cv::DMatch> _LRmatches, cv::Mat _rawImg) {
            frameID = _frameID;
            pose = _pose;
            keypoints = _keypoints;
            descriptors = _descriptors;
            intrinsics = _intrinsics;
            rightFrame = nullptr;
            LRmatches = _LRmatches;
            rawImg = _rawImg;
            // inlierflag
            featureInlierFlag.resize(keypoints.size(), true);
        }

        Sophus::SE3d getPose() {
            return pose;
        }

        /*
        This is to get the pose of the right frame wrt the left frameo or if seen from the left frame what is 
        the pose of the right frame.
        */
        Sophus::SE3d getRightPoseInWorldFrame() {
            std::unique_lock<std::mutex> lock(poseMutex);

            // get sophus matrix from Eigen
            Eigen::Matrix4d poseMat;
            poseMat << 1, 0, 0, -intrinsics->Right.getBaseline(), 
                       0, 1, 0, 0,
                       0, 0, 1, 0, 
                       0, 0 ,0 ,1;

            Sophus::SE3d returnPose(this->pose.matrix()*poseMat);

            return returnPose;

        }

        void setPose(Sophus::SE3d _pose) {
            std::unique_lock<std::mutex> lock(poseMutex);

            this->pose = _pose;
        }

        unsigned long getFrameID() {
            std::unique_lock<std::mutex> lock(poseMutex);

            return frameID;
        }

        void clearKeypoints() {
            this->keypoints = {};
        }

        void clearInlierFlags() {
            this->featureInlierFlag = {};
        }

        void addKeypoint(const cv::KeyPoint kp) {
            std::unique_lock<std::mutex> lock(poseMutex);
            this->keypoints.push_back(kp);  
        }
        
        void setLRmatches(std::vector<cv::DMatch> &matches) {
            std::unique_lock<std::mutex> lock(poseMutex);
            this->LRmatches = matches;
        }

        bool resetFeatureInliers() {
            std::unique_lock<std::mutex> lock(poseMutex);
            featureInlierFlag.clear();
            featureInlierFlag.resize(keypoints.size(), true);
            return true;
        }
        auto getAllInliers() {
            std::unique_lock<std::mutex> lock(poseMutex);
            return featureInlierFlag;
        }
        bool getFeatureInlierFlag(int kpID) {
            std::unique_lock<std::mutex> lock(poseMutex);
            return featureInlierFlag[kpID];
        }

        bool setFeatureInlierFlag(int kpID, bool flag) {
            std::unique_lock<std::mutex> lock(poseMutex);
            if (kpID >= featureInlierFlag.size()) {
                // featureInlierFlag.resize(kpID+10, true);
                featureInlierFlag.emplace_back(true);
                return true;
            }
            featureInlierFlag[kpID] = flag;
            return true;
        }

        // set all inliers to a value
        bool setAllInliers(bool flag) {
            std::unique_lock<std::mutex> lock(poseMutex);
            if (featureInlierFlag.size() == 0) {
                LOG(ERROR) << "Frame ID: " << frameID << " does not have any features FLAGS";
                featureInlierFlag.resize(keypoints.size(), flag);
                return true;
            }
            for (int i = 0; i < featureInlierFlag.size(); i++) {
                featureInlierFlag[i] = flag;
            }
            return true;
        }

        bool updateMatchesMap (unsigned long _frameID, std::vector<cv::DMatch> &matches) {
            std::unique_lock<std::mutex> lock(poseMutex);
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

        static unsigned long createFrameID() {
            static unsigned long frameID = 0;
            return frameID++;
        }


        // add observation/3D points to the frame
        void addObservation(MapPoint::Ptr mapPoint) {
            std::unique_lock<std::mutex> lock(poseMutex);
            if (obsMapPoints.find(mapPoint->getMapPointID()) != obsMapPoints.end()) {
                LOG(ERROR) << "Frame ID: " << frameID << " already has a map point ID: " << mapPoint->getMapPointID();
                return;
            }
            obsMapPoints[mapPoint->getMapPointID()] = mapPoint;
        }

        cv::Mat getRawImg() {
            std::unique_lock<std::mutex> lock(poseMutex);
            return rawImg;
        }

        cv::Mat getDescriptor() {
            std::unique_lock<std::mutex> lock(poseMutex);
            return descriptors;
        }

        std::vector<cv::DMatch> getLRMatches() {
            std::unique_lock<std::mutex> lock(poseMutex);
            return LRmatches;
        }

        std::vector<cv::KeyPoint> getKeypoints() {
            std::unique_lock<std::mutex> lock(poseMutex);
            return keypoints;
        }

        // Gets the MapPointID from the keypoint ID
        int getMpIDfromKpID(int kpID) {
            std::unique_lock<std::mutex> lock(poseMutex);
            for (auto &obsMapPoint : obsMapPoints) {
                if (obsMapPoint.second->getKpID(this->frameID) == kpID) {
                    return obsMapPoint.first;
                }
            }
            // LOG(ERROR) << "Frame ID: " << frameID << " does not have a map point ID for keypoint ID: " << kpID;
            return -1;
        }

        // set the mp ID for a keypoint ID
        bool setMpIDforKpID(int kpID, int mpID) {
            std::unique_lock<std::mutex> lock(poseMutex);
            for (auto &obsMapPoint : obsMapPoints) {
                if (obsMapPoint.second->getKpID(this->frameID) == kpID) {
                    obsMapPoint.second->setMapPointID(mpID);
                    return true;
                }
            }
            LOG(ERROR) << "Frame ID: " << frameID << " does not have a map point ID for keypoint ID: " << kpID;
            return false;
        }



        // get descriptors
        cv::Mat getDescriptors() {
            std::unique_lock<std::mutex> lock(poseMutex);
            return this->descriptors;
        }




        std::map<unsigned long, MapPoint::Ptr> getObsMapPoints() {
            std::unique_lock<std::mutex> lock(poseMutex);
            return obsMapPoints;
        }

        cv::Point2f world2pixel(const Vec3 &pWorld, const Intrinsics::Ptr intrinsics) {
            Eigen::Matrix<double, 4, 1> homo3DPt;
            homo3DPt << pWorld.coeff(0), pWorld.coeff(1), pWorld.coeff(2), 1;
            Eigen::Matrix3d K;
            intrinsics->Left.getKEigen(K);
            Vec3 camPoints = K*this->pose.matrix3x4()*homo3DPt;

            return cv::Point2f(camPoints.coeff(0)/camPoints.coeff(2), camPoints.coeff(1)/camPoints.coeff(2));
        }
};

#endif