#include "../include/Utils.hpp"
#include <opencv4/opencv2/cudafeatures2d.hpp>


enum DetectorType {
    ORB,
    SIFT,
    SURF,
    FAST,
    BRISK
};


enum DescriptorType {
    BRIEF
};


template <typename T>
class Features {
    private:
        DetectorType detectorType;
        DescriptorType descriptorType;
        cv::Ptr<cv::FeatureDetector> detector;
        cv::Ptr<cv::cuda::ORB> detectorGPU;
        cv::Ptr<cv::cuda::DescriptorMatcher> matcher;
        bool isGPU;
        bool initDetector() {
            try {
                if (detectorType == ORB) {
                    if (isGPU) {
                        detectorGPU = cv::cuda::ORB::create();
                    }
                    else {
                        detector = cv::ORB::create();
                    }
                }
                else {
                    LOG(ERROR) << "Feature Detector: " << detectorType << "not implemented";
                    return false;
                }
            } catch (cv::Exception &e) {
                std::cout << "Exception in feature Detection" << std::endl;
                LOG(ERROR) << "Exception in Initializing feature Detection";
            }
        }
    public:
        typedef std::shared_ptr<Features> Ptr;
        Features(DetectorType _detectorType, DescriptorType _descType, bool _isGPU = false) {
            detectorType = _detectorType;
            descriptorType = _descType;
            isGPU = _isGPU;

        }
        
        void detectFeaturesGPU(T &img, std::vector< &keypoints, cv::cuda::GpuMat &descriptors) {
            detectorGPU.empty();
            detectorGPU->detectAndCompute(img, cv::cuda::GpuMat(), keypoints, descriptors);
        }
        
        void matchFeaturesGPU(cv::cuda::GpuMat &descriptors1, cv::cuda::GpuMat &descriptors2, std::vector<cv::DMatch> &matches) {
            matcher->match(descriptors1, descriptors2, matches);
        }
    


};