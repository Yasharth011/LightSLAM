#include "LighterGlue.h"
#include "OnnxHelper.h"

Lighterglue::Lighterglue(const std::string &modelFile) {
    // Convert the modelFile path to onnx compatible path
    std::vector<ORTCHAR_T> modelFileOrt;
    OnnxHelper::Str2Ort(modelFile, modelFileOrt);

    // create onnx runtime session
    ortEnv_ = std::unique_ptr<Ort::Env>(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Lighterglue"));

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    ortSession_ = std::unique_ptr<Ort::Session>(new Ort::Session(*ortEnv_, modelFileOrt.data(), sessionOptions));

    // Get model info
    OnnxHelper::GetModelInfo(*ortSession_, inputInfos_, outputInfos_);

    // set input and output names
    for (const auto &inputInfo : inputInfos_) {
        inputNames_.push_back(inputInfo.name.c_str());
    }
    for (const auto &outputInfo : outputInfos_) {
        outputNames_.push_back(outputInfo.name.c_str());
    }
    OnnxHelper::PrintModelInfo(inputInfos_, outputInfos_);
}

void Lighterglue::Match( std::vector<cv::KeyPoint>& kpts0,  cv::Mat& desc0, cv::Size imgSize0,
         std::vector<cv::KeyPoint>& kpts1,  cv::Mat& desc1, cv::Size imgSize1,
        std::vector<cv::DMatch>& matches){

            cv::Mat normkpts0 , normkpts1 ;
            normkpts0 = Norm_kpts( kpts0, imgSize0);

            normkpts1 = Norm_kpts( kpts1, imgSize1);

            // 2. Count how many points we actually have right now
            int64_t num_kpts0 = static_cast<int64_t>(kpts0.size());
            int64_t num_kpts1 = static_cast<int64_t>(kpts1.size());

            // 3. Construct the real 3D shapes substituting the actual counts
            std::vector<int64_t> kpts0Shape = {1, num_kpts0, 2};
            std::vector<int64_t> desc0Shape = {1, num_kpts0, 64}; // 64 is the descriptor size from XFeat
            
            std::vector<int64_t> kpts1Shape = {1, num_kpts1, 2};
            std::vector<int64_t> desc1Shape = {1, num_kpts1, 64};

            // 4. Create the input tensor container for ONNX Runtime
            std::vector<Ort::Value> inputTensors;

            // Image 0 inputs
            //  CORRECT ORDER (Matches ONNX input requirements)
// 1. kpts0
            inputTensors.emplace_back(OnnxHelper::CreateTensor<float>(kpts0Shape, normkpts0.ptr<float>(), normkpts0.total(), true));

            // 2. kpts1
            inputTensors.emplace_back(OnnxHelper::CreateTensor<float>(kpts1Shape, normkpts1.ptr<float>(), normkpts1.total(), true));

            // 3. desc0
            inputTensors.emplace_back(OnnxHelper::CreateTensor<float>(desc0Shape, desc0.ptr<float>(), desc0.total(), true));

            // 4. desc1
            inputTensors.emplace_back(OnnxHelper::CreateTensor<float>(desc1Shape, desc1.ptr<float>(), desc1.total(), true));

            auto outputTensors = ortSession_->Run(Ort::RunOptions{nullptr}, inputNames_.data(),
                                          inputTensors.data(), inputTensors.size(), outputNames_.data(), outputNames_.size());

            auto* kptmatches = outputTensors[0].GetTensorMutableData<int64_t>();
            auto* kptscores = outputTensors[1].GetTensorMutableData<float>();

            
            Ort::TensorTypeAndShapeInfo info = outputTensors[0].GetTensorTypeAndShapeInfo();
            std::vector<int64_t> shape = info.GetShape();//This returns a vector of size containing the num_matches and two

            // 2. Extract num_matches (it's the first dimension / number of rows)
            int64_t num_matches = shape[0]; 

            for(int i = 0; i < num_matches; i++  ){
                int64_t idx0 = kptmatches[i*2+0];
                int64_t idx1 = kptmatches[i*2+1];

                matches.emplace_back(static_cast<int>(idx0), static_cast<int>(idx1), 1.0f - kptscores[i]);
            }

        }
        
cv::Mat Lighterglue::Norm_kpts(std::vector<cv::KeyPoint>& kpts, cv::Size imgSize) {
    cv::Mat normKpts(kpts.size(), 2, CV_32F);
    float H = imgSize.height;
    float W = imgSize.width;

    for (int i = 0; i < kpts.size(); i++) {
        normKpts.at<float>(i, 0) = ((2.0f * kpts[i].pt.x) / (W - 1.0f)) - 1.0f;
        normKpts.at<float>(i, 1) = ((2.0f * kpts[i].pt.y) / (H - 1.0f)) - 1.0f;
    }

    return normKpts;
}
