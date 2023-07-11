#include "../include/3DHandler.hpp"




bool _3DHandler::getEssentialMatrix(std::vector<cv::DMatch> &matches, Frame::Ptr srcFrame, Frame::Ptr dstFrame, cv::Mat &F) {
    std::vector<cv::Point2f> srcPts;
    std::vector<cv::Point2f> dstPts;
    for (auto &match : matches) {
        srcPts.push_back(srcFrame->keypoints[match.queryIdx].pt);
        dstPts.push_back(dstFrame->keypoints[match.trainIdx].pt);
    }

    double focal = this->intrinsics->Left.getF();
    double cx = this->intrinsics->Left.getCx();
    double cy = this->intrinsics->Left.getCy();
    cv::Point2d principalPoint(cx, cy);

    // WARNING : Essential matrix is from dstFrame to srcFrame
    // this way we get pose of frame 2 in frame 1
    try {    
        E = cv::findEssentialMat(dstPts, srcPts, focal, principalPoint, cv::RANSAC, 0.999, 1.0, mask);
    } catch (const std::exception &e) {
        LOG(ERROR) << "Exception in findEssentialMat: " << e.what();
        LOG(ERROR) << "srcPts: " << srcPts.size() << " dstPts: " << dstPts.size();
        LOG(ERROR) << "srcFrame->keypoints: " << srcFrame->keypoints.size() << " dstFrame->keypoints: " << dstFrame->keypoints.size();
        return false;
    }
    return true;
}



bool _3DHandler::getPoseFromEssential(const cv::Mat &E, const std::vector<cv::DMatch> &matches, Frame::Ptr srcFrame, Frame::Ptr dstFrame, Pose &pose) {
    std::vector<cv::Point2f> srcPts;
    std::vector<cv::Point2f> dstPts;
    for (auto &match : matches) {
        srcPts.push_back(srcFrame->keypoints[match.queryIdx].pt);
        dstPts.push_back(dstFrame->keypoints[match.trainIdx].pt);
    }

    double focal = this->intrinsics->Left.getF();
    double cx = this->intrinsics->Left.getCx();
    double cy = this->intrinsics->Left.getCy();
    cv::Point2d principalPoint(cx, cy);

    cv::Mat R, t;
    int inliers;
    try {
        inliers = cv::recoverPose(E, dstPts, srcPts, instrinsics->Left.getK(), R, t);
    } catch (const std::exception &e) {
        LOG(ERROR) << "Exception in recoverPose: " << e.what();
        LOG(ERROR) << "srcPts: " << srcPts.size() << " dstPts: " << dstPts.size();
        LOG(ERROR) << "srcFrame->keypoints: " << srcFrame->keypoints.size() << " dstFrame->keypoints: " << dstFrame->keypoints.size();
        return false;
    }
    pose = Pose(R, t, this->intrinsics->Left.getK());    
    
    return true;
}

// trinagulation after creating Map and MapPoints

bool _3DHandler::triangulateAll(Frame::Ptr srcFrame, Frame::Ptr dstFrame, const std::vector<cv::DMatch> &matches, cv::Mat &pnts3D) {
    
    // set poses for both views
    std::vector<Sophus::SE3d> poses{srcFrame->pose, dstFrame->pose};

    std::vector<cv::Point2f> srcPts;
    std::vector<cv::Point2f> dstPts;

    int landmarkCount = 0;
    for (auto &match : matches) {
        // get the keypoints that have been matched
        std::vector<Vec3> points {
            this->intrinsics->Left.pixel2camera(cv::Point(srcFrame->keypoints[match.queryIdx].pt));
            this->intrinsics->Left.pixel2camera(cv::Point(dstFrame->keypoints[match.trainIdx].pt));
        }
        Vec3 pWorld = Vec3::Zero();
        if (triangulatePoint(poses, points, pWorld) && pWorld[2] > 0) {
            // update landmark count
            landmarkCount++;
            // add the 3d point to the map
            auto newMapPoint = std::make_shared<MapPoint>(MapPoint::createMapPointID(), pWorld);
            // register the features that led to creation of this 3d point
            newMapPoint->addObservation(srcFrame->getFrameID(), match.queryIdx);
            newMapPoint->addObservation(dstFrame->getFrameID(), match.trainIdx);

            // add the 3d point to the frame observations
            srcFrame->addMapPoint(newMapPoint);
            dstFrame->addMapPoint(newMapPoint);

            // add the 3d point to the map itself??????*******************  
        }          
    }
    LOG(INFO) << "Triangulated: " << landmarkCount << " points";
    LOG(INFO) << "Total Matches: " << matches.size();
}




inline bool _3DHandler::triangulatePoint(const std::vector<Sophus::SE3d> &poses,
                   const std::vector<Vec3> points, Vec3 &3DPoint) {
    MatXX A(2 * poses.size(), 4);
    VecX b(2 * poses.size());
    b.setZero();
    for (size_t i = 0; i < poses.size(); ++i) {
        Mat34 m = poses[i].matrix3x4();
        A.block<1, 4>(2 * i, 0) = points[i][0] * m.row(2) - m.row(0);
        A.block<1, 4>(2 * i + 1, 0) = points[i][1] * m.row(2) - m.row(1);
    }
    auto svd = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV);
    3DPoint = (svd.matrixV().col(3) / svd.matrixV()(3, 3)).head<3>();

    if (svd.singularValues()[3] / svd.singularValues()[2] < 1e-2) {
        // 解质量不好，放弃
        return true;
    }
    return false;
}
