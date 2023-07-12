#ifndef FEATURES_HPP
#define FEATURES_HPP


#include "../include/Utils.hpp"
#include <opencv4/opencv2/cudafeatures2d.hpp>
#include <opencv4/opencv2/features2d.hpp>

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
        cv::Ptr<cv::FeatureDetector> detectorCPU;
        cv::Ptr<cv::cuda::ORB> detectorGPU;
        cv::Ptr<cv::cuda::DescriptorMatcher> matcherGPU;
        cv::Ptr<cv::DescriptorMatcher> matcherCPU;

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
                        detectorCPU = cv::ORB::create(500);
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
                if (isGPU) {
                    matcherGPU = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
                    LOG(INFO) << "Matcher: " << DescriptorType::BRIEF << " initialized with CUDA";
                    std::cout << "Matcher: " << DescriptorType::BRIEF << " initialized with CUDA" << std::endl;
                } else {
                    matcherCPU = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
                    LOG(INFO) << "Matcher: " << DescriptorType::BRIEF << " initialized";
                    std::cout << "Matcher: " << DescriptorType::BRIEF << " initialized" << std::endl;
                }
            } catch(cv::Exception &e) {
                LOG(ERROR) << "Exception in Initializing feature Matcher";
                std::cout << "Exception in Initializing feature Matcher" << std::endl;
            }

        }
    public:
        Features() {}
        Features(DetectorType _detectorType, DescriptorType _descType) {
            detectorType = _detectorType;
            descriptorType = _descType;
            // get cuda device count
            bool cudaSupported = checkCUDAsupport();
            if (!cudaSupported) {
                LOG(ERROR) << "No CUDA enabled devices found";
                std::cout << "No CUDA enabled devices found" << std::endl;
                isGPU = false;
            }
            else {
                LOG(INFO) << "CUDA enabled devices found";
                std::cout << "CUDA enabled devices found" << std::endl;
                isGPU = true;
            }
            init();

        }
        
        void detectFeatures(T &img, cv::cuda::GpuMat &keypoints, cv::cuda::GpuMat &descriptors) {
           
            detectorGPU->detectAndComputeAsync(img, cv::cuda::GpuMat(), keypoints, descriptors);
        }

        void detectFeatures(T &img, std::vector<cv::KeyPoint> &keypoints, cv::Mat &descriptors) {
            detectorCPU->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
        }
        
        void matchFeatures(cv::cuda::GpuMat &descriptors1, cv::cuda::GpuMat &descriptors2, std::vector<cv::DMatch> &matches) {
            matcherGPU->match(descriptors1, descriptors2, matches);
        }

        void matchFeatures(cv::Mat &descriptors1, cv::Mat &descriptors2, std::vector<cv::DMatch> &matches) {
            matcherCPU->match(descriptors1, descriptors2, matches);
        }

        void drawMatches(T &img1, T &img2, cv::cuda::GpuMat &keypoints1, cv::cuda::GpuMat &keypoints2, std::vector<cv::DMatch> &matches, cv::Mat &outImg) {
            std::vector<cv::KeyPoint> kp1CPU, kp2CPU;
            cv::Mat Img1CPU, Img2CPU;
            img1.download(Img1CPU);
            img2.download(Img2CPU);
            detectorGPU->empty();
            detectorGPU->convert(keypoints1, kp1CPU);
            detectorGPU->convert(keypoints2, kp2CPU);
            cv::drawMatches(Img1CPU, kp1CPU, Img2CPU, kp2CPU, matches, outImg);
        }

        void drawMatches(T img1, T img2, std::vector<cv::KeyPoint> &keypoints1, std::vector<cv::KeyPoint> &keypoints2, std::vector<cv::DMatch> &matches, cv::Mat &outImg) {
            cv::drawMatches(img1, keypoints1, img2, keypoints2, matches, outImg);
        }

        void removeOutliers(std::vector<cv::DMatch> &matches, std::vector<cv::DMatch> &outMatches) {
            double max_dist = 0; double min_dist = 100;
            //-- Quick calculation of max and min distances between keypoints
            auto result = std::minmax_element(matches.begin(), matches.end(), [](const cv::DMatch &a, const cv::DMatch &b) {
                    return a.distance < b.distance;});

            // Remove matches with distance greater than 2*min_dist

            for (auto &eachMatch : matches) {
                if (eachMatch.distance < 3*result.first->distance){

                    outMatches.push_back(eachMatch);
                }
            }
        }

        void convertGPUKpts(std::vector<cv::KeyPoint> &keypoints, cv::cuda::GpuMat &keypointsGPU) {
            detectorGPU->convert(keypointsGPU, keypoints);
        }
        typedef std::shared_ptr<Features<T>> Ptr;

};


#endif