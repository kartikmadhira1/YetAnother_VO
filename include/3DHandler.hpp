#ifndef _3DHANDLER_H
#define _3DHANDLER_H

#include "Utils.hpp"
#include "Frame.hpp"
#include "opencv2/calib3d.hpp"
#include "Camera.hpp"

struct Pose {
    private:
        cv::Mat R;
        cv::Mat t;
        cv::Mat P;
        cv::Mat _3DPts;
        cv::Mat _2DPts;
        Sophus::SE3d sophusPose;
        int numChierality = 0;
    public:
        Pose(cv::Mat _R, cv::Mat _t, cv::Mat P) {
            this->R = _R;
            this->t = _t;
            this->P = P;
            
            Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic> eigenR;
            cv::cv2eigen(_R, eigenR);
            Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic> eigenT;
            cv::cv2eigen(_t, eigenT);
            this->sophusPose = Sophus::SE3d(eigenR, eigenT);
        }
       
        Pose() {}
        Pose operator=(const Pose &pose) {
            this->R = pose.R;
            this->t = pose.t;
            this->P = pose.P;
            this->numChierality = pose.numChierality;
            Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic> eigenR;
            cv::cv2eigen(this->R, eigenR);
            Eigen::Matrix<double,Eigen::Dynamic, Eigen::Dynamic> eigenT;
            cv::cv2eigen(this->t, eigenT);
            this->sophusPose = Sophus::SE3d(eigenR, eigenT);
            return *this;
        }
        
        Sophus::SE3d getPose() {
            return this->sophusPose;
        }
        int getNumChierality() {
            return this->numChierality;
        }
};



class _3DHandler {
    private:
        Instrincics::Ptr instrinsics;
    public:
        _3DHandler(Instrinsics::Ptr &_instrinsics) {
            this->instrinsics = _instrinsics;
        }

        /*
            * @brief: Given a set of matches between src and dst Frames, compute the fundamental matrix
            * @param: matches: vector of matches of type cv::DMatch between two frames
            * @param: srcFrame: source frame of type Frame::Ptr
            * @param: dstFrame: destination frame of type Frame::Ptr
            * @param: E: fundamental matrix of type cv::Mat
            * @return: bool: true if fundamental matrix is computed successfully, false otherwise
        */
        bool getEssentialMatrix(const std::vector<cv::DMatch> &matches, Frame::Ptr srcFrame, Frame::Ptr dstFrame, cv::Mat &E);

        /*
            *@brief: Given set of matches, compute the pose of the destination frame with respect to the source frame
            *@param: E: essential matrix of type cv::Mat
            *@param: matches: vector of matches of type cv::DMatch between two frames
            *@param: srcFrame: source frame of type Frame::Ptr
            *@param: dstFrame: destination frame of type Frame::Ptr
            *@param: pose: pose of the destination frame with respect to the source frame of type Pose
            *@return: bool: true if pose is computed successfully, false otherwise
        */
        bool getPoseFromEssential(const cv::Mat &E, const std::vector<cv::DMatch> &matches, Frame::Ptr srcFrame, Frame::Ptr dstFrame, Pose &pose);
        
        /*
            *@brief: Linear Triangulation of all 3D points from two views
            *@param: srcFrame: source frame of type Frame::Ptr
            *@param: dstFrame: destination frame of type Frame::Ptr
            *@param: matches: vector of matches of type cv::DMatch between two frames
            *@param: pnts3D: 3D points of type cv::Mat
            *@return: bool: true if 3D points are computed successfully, false otherwise
        */
        bool triangulateAll(Frame::Ptr srcFrame, Frame::Ptr dstFrame, const std::vector<cv::DMatch> &matches, cv::Mat &pnts3D);

        /*
            *@brief: Linear Triangulation of single 3d point from two views
            *@param: poses : vector of poses of type Sophus::SE3d, pose for each view
            *@param: points: left and right points of type Vec3, stored as vector
            *@param: 3DPoint: Triangulated 3D point of type Vec3
            *@return: bool: true if 3D point is computed successfully, false otherwise
        
        */
        bool triangulePoint(const std::vector<Sophus::SE3d> &poses,
                   const std::vector<Vec3> lrPoints, Vec3 &3DPoint);
        
        
        
        
        
        
        
        
        
        
        
        
        // bool getFRANSAC(std::vector<Matches> matches, cv::Mat &F, 
        //               int iterations, double threshold);
        // Pose disambiguateRT(const cv::Mat &E, std::vector<Matches> &matches);

        // cv::Mat constructNormMatrix(std::vector<double> xVec, std::vector<double> yVec, 
        //                                         double xMean, double yMean);
        // bool checkDepthPositive(cv::Mat &pnts3D, cv::Mat R1, cv::Mat R2, cv::Mat t1, cv::Mat t2, Pose &pose);
        // float unNormalizePoint(float pt, float mean, float var);
        // cv::Mat rotateMatrixZ(int rotateAngle);
        // double getMeanVar(std::vector<double> &vec);

        ~_3DHandler();
};

#endif // TODOITEM_H
