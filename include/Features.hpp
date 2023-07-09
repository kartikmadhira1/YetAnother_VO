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
        cv::Ptr<cv::cuda::DescriptorMatcher> matcherGPU;
        bool isGPU;
        bool init() {
            try {
                if (detectorType == DetectorType::ORB) {
                    if (isGPU) {
                        detectorGPU = cv::cuda::ORB::create(500);
                        detectorGPU->setBlurForDescriptor(true);
                        LOG(INFO) << "ORB Detector initialized with CUDA";
                        std::cout << "ORB Detector initialized with CUDA" << std::endl;
                    }
                    else {
                        detector = cv::ORB::create();
                        LOG(INFO) << "ORB Detector initialized";
                        std::cout << "ORB Detector initialized" << std::endl;
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

            // Initialize Matcher
            try {
                matcherGPU = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
                LOG(INFO) << "Matcher: " << DescriptorType::BRIEF << " initialized with CUDA";
                std::cout << "Matcher: " << DescriptorType::BRIEF << " initialized with CUDA" << std::endl;
            } catch(cv::Exception &e) {
                LOG(ERROR) << "Exception in Initializing feature Matcher";
                std::cout << "Exception in Initializing feature Matcher" << std::endl;
            }

        }
    public:
        typedef std::shared_ptr<Features> Ptr;
        Features(DetectorType _detectorType, DescriptorType _descType, bool _isGPU = false) {
            detectorType = _detectorType;
            descriptorType = _descType;
            isGPU = _isGPU;
            init();

        }
        
        void detectFeaturesGPU(T &img, cv::cuda::GpuMat &keypoints, cv::cuda::GpuMat &descriptors) {
            detectorGPU.empty();
            int ngpus = cv::cuda::getCudaEnabledDeviceCount();
            // LOG(INFO) << "Number of GPUs: " << ngpus;
            // std::cout << "Number of GPUs: " << ngpus << std::endl;
            detectorGPU->detectAndComputeAsync(img, cv::cuda::GpuMat(), keypoints, descriptors);
        }
        
        void matchFeaturesGPU(cv::cuda::GpuMat &descriptors1, cv::cuda::GpuMat &descriptors2, std::vector<cv::DMatch> &matches) {
            matcherGPU->match(descriptors1, descriptors2, matches);
        }

        void drawMatchesGPU(T &img1, T &img2, cv::cuda::GpuMat &keypoints1, cv::cuda::GpuMat &keypoints2, std::vector<cv::DMatch> &matches, cv::Mat &outImg) {
            std::vector<cv::KeyPoint> kp1CPU, kp2CPU;
            cv::Mat Img1CPU, Img2CPU;
            img1.download(Img1CPU);
            img2.download(Img2CPU);
            detectorGPU->empty();
            std::cout << "kp1" << keypoints1.size() << std::endl;
            std::cout <<  "kp2 "<<  keypoints2.size() << std::endl;
            std::cout << kp1CPU.size() << std::endl;
            std::cout << kp2CPU.size() << std::endl;
            detectorGPU->convert(keypoints1, kp1CPU);
            detectorGPU->convert(keypoints2, kp2CPU);

            cv::drawMatches(Img1CPU, kp1CPU, Img2CPU, kp2CPU, matches, outImg);
        }


};