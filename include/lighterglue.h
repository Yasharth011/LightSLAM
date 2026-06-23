#pragma once

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "OnnxHelper.h"


class Lighterglue {
public:
    explicit Lighterglue(const std::string &modelFile);

    void Match( std::vector<cv::KeyPoint>& kpts0,  cv::Mat& desc0, cv::Size imgSize0,
         std::vector<cv::KeyPoint>& kpts1,  cv::Mat& desc1, cv::Size imgSize1,
        std::vector<cv::DMatch>& matches);

private:
    cv::Mat Norm_kpts(std::vector<cv::KeyPoint>& kpts , cv::Size imgSize);


private:
    std::unique_ptr<Ort::Env> ortEnv_;
    std::unique_ptr<Ort::Session> ortSession_;

    // input and output infos
    std::vector<TensorInfo> inputInfos_;
    std::vector<TensorInfo> outputInfos_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;



};
