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