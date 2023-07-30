#ifndef OPTFLOW_HPP
#define OPTFLOW_HPP

#include "Utils.hpp"
#include "Frame.hpp"
#include <opencv4/opencv2/cudaoptflow.hpp>
#include <opencv4/opencv2/optflow.hpp>



class OptFlow {

    private:
        cv::Ptr<cv::cuda::SparsePyrLKOpticalFlow> optFlowGPU;
        cv::Ptr<cv::SparsePyrLKOpticalFlow> optFlowCPU;
        // prevPts, nextPts, status, err;
        std::vector<uchar> flowStatus;
        cv::Mat error;

        bool isGPU;
    public:
        typedef std::shared_ptr<OptFlow> Ptr;

        OptFlow() {
            bool cudaSupported = checkCUDAsupport();
            if (!cudaSupported) {
                LOG(ERROR) << "Optical flow with GPU not supported";
                isGPU = false;
            }
            else {
                LOG(INFO) << "Optical flow with GPU supported";
                isGPU = true;
            }
            init();
        }

        void init() {
            if (isGPU) {
                optFlowGPU = cv::cuda::SparsePyrLKOpticalFlow::create();
                optFlowGPU->setWinSize(cv::Size(11, 11));
                // optFlowGPU->
            }
            else {
                optFlowCPU = cv::SparsePyrLKOpticalFlow::create();
            }
        }

        bool getOptFlow(Frame::Ptr srcFrame, Frame::Ptr dstFrame, std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts) {
            if (isGPU) {
                // convert all images to GPU
                cv::cuda::GpuMat srcImg, dstImg, flowStatusGPU, errorGPU;
                cv::cuda::GpuMat gpuPrevPts;
                cv::cuda::GpuMat gpuNextPts;
                srcImg.upload(srcFrame->getRawImg());
                dstImg.upload(dstFrame->getRawImg());
                if(convertToGpuArray(prevPts, nextPts, gpuPrevPts, gpuNextPts)) {
                    try {
                        optFlowGPU->calc(srcImg, dstImg, gpuPrevPts, gpuNextPts, flowStatusGPU, errorGPU);
                    } catch (const cv::Exception& e) {
                        LOG(ERROR) << "Could not calculate optical flow between frame ID: " << srcFrame->getFrameID() << " and frame ID: " << dstFrame->getFrameID() << " with error: " << e.what();
                        LOG(ERROR) << "Exception caught: " << e.what();
                        return false;
                    }
                    downloadToCpu(flowStatusGPU, flowStatus);
                }
                return true;
            } else {
                LOG(ERROR) << "Optical flow with CPU not implemented yet";
                return false;
            }
            return true;
        }

        bool convertToGpuArray(std::vector<cv::Point2f> &prevPts, std::vector<cv::Point2f> &nextPts, cv::cuda::GpuMat &gpuPrevPts, cv::cuda::GpuMat &gpuNextPts) {
            if (isGPU) {
                cv::Mat mat1(1, prevPts.size(), CV_32FC2, prevPts.data());
                cv::Mat mat2(1, nextPts.size(), CV_32FC2, nextPts.data());
                gpuPrevPts.upload(mat1);
                gpuNextPts.upload(mat2);
                return true;
            } else {
                LOG(ERROR) << "Optical flow with CPU not implemented yet";
                return false;
            }
            return true;
        }

        void downloadToCpu(const cv::cuda::GpuMat& dMat, std::vector<uchar>& vec)
        {
            vec.resize(dMat.cols);
            cv::Mat mat(1, dMat.cols, CV_8UC1, (void*)&vec[0]);
            dMat.download(mat);
        }

        std::vector<uchar> getFlowStatus() {
            return flowStatus;
        }


};

#endif