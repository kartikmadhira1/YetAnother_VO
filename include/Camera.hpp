#ifndef CAMERA_H
#define CAMERA_H

#include "Utils.hpp"


struct Camera {
    public:
        Camera() {

        }
        Camera(const cv::Mat &_cam) {
            _cam.copyTo(fullMatrix);
            K = fullMatrix(cv::Range(0, 3), cv::Range(0, 3));
            P = fullMatrix(cv::Range(0, 4), cv::Range(0, 4));

        }
        cv::Mat K;
        cv::Mat P;

        cv::Mat fullMatrix;
        cv::Mat getK() {
            return K;
        }

        bool getKEigen(Eigen::Matrix3d &K) {
            try {
                cv::cv2eigen(this->K, K);
            } catch (const cv::Exception& e) {
                LOG(ERROR) << "Could not convert K to Eigen matrix with error: " << e.what();
                LOG(ERROR) << "Exception caught: " << e.what();
                return false;
            }
            return true;
        }


        cv::Mat getP() {
            return P;
        }
        float getFx() {
            return K.at<double>(0,0);
        }
        float getFy() {
            return K.at<double>(1,1);
        }
        float getF() {
            return (getFx() + getFy())/2;
        }
        float getCx() {
            return K.at<double>(0,2);
        }
        float getCy() {
            return K.at<double>(1,2);
        }
        double getBaseline() {
            double baseline = -P.at<double>(0,3)/P.at<double>(0,0);
            return baseline;
        }
        void getCalibMat() {}
        
        Vec3 pixel2camera(const cv::Point p) {
            double depth = 1;
            return Vec3(
            (p.x - getCx()) * depth / getFx(),
            (p.y - getCy()) * depth / getFy(),
            1);
        }
        
        void printK() {
            for (int i=0; i < K.rows; i++) {
                for (int j=0; j<K.cols;j++) {
                    std::cout <<  K.at<double>(i, j) << " ";
                }
                std::cout <<  "\n" << std::endl;
            }
        }
        void printP() {
            for (int i=0; i < P.rows; i++) {
                for (int j=0; j < P.cols;j++) {
                    std::cout <<  P.at<double>(i, j) << " ";
                }
                std::cout <<  "\n" << std::endl;
            }
        }
        ~Camera() {
        }
};


struct Intrinsics {
    public:
        Intrinsics() {

        }
        typedef std::shared_ptr<Intrinsics> Ptr;
        Camera Left;
        Camera Right;

        ~Intrinsics() {

        }
};

#endif // TODOITEM_H