#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

#include "LighterGlue.h"
#include "XFeat.h"

int main(int argc, char **argv) {
  // Parse arguments (Fixed missing semicolon inside argument keys)
  const std::string argKeys =
      "{model1 | ../model/xfeat_640x640.onnx | model file path}"
      "{img1 | ../data/1.png | the first image file path}"
      "{img2 | ../data/2.png | the second image file path}"
      "{model2 | ../model/lighterglue_L3.onnx | model file path};";

  cv::CommandLineParser parser(argc, argv, argKeys);
  auto model1File = parser.get<std::string>("model1");
  auto imgFile1 = parser.get<std::string>("img1");
  auto imgFile2 = parser.get<std::string>("img2");
  auto model2File = parser.get<std::string>("model2");

  std::cout << "model file 1: " << model1File << std::endl;
  std::cout << "model file 2: " << model2File << std::endl;
  std::cout << "image file 1: " << imgFile1 << std::endl;
  std::cout << "image file 2: " << imgFile2 << std::endl;

  std::cout << "creating XFeat...\n";
  XFeat xfeat(model1File);
  std::cout << "creating LighterGlue...\n";
  Lighterglue lighterglue(model2File);

  std::cout << "reading images...\n";
  cv::Mat img1 = cv::imread(imgFile1, cv::IMREAD_GRAYSCALE);
  cv::Mat img2 = cv::imread(imgFile2, cv::IMREAD_GRAYSCALE);

  cv::Size img1_size = img1.size();
  cv::Size img2_size = img2.size();

  // Extract features
  std::cout << "detecting features ...\n";
  std::vector<cv::KeyPoint> keys1, keys2;
  cv::Mat descs1, descs2;
  xfeat.DetectAndCompute(img1, keys1, descs1, 2048);
  xfeat.DetectAndCompute(img2, keys2, descs2, 2048);

  // Visualize keypoints
  cv::Mat imgColor1, imgColor2;
  cv::cvtColor(img1, imgColor1, cv::COLOR_GRAY2BGR);
  cv::cvtColor(img2, imgColor2, cv::COLOR_GRAY2BGR);
  cv::drawKeypoints(imgColor1, keys1, imgColor1, cv::Scalar(0, 0, 255));
  cv::drawKeypoints(imgColor2, keys2, imgColor2, cv::Scalar(0, 0, 255));
  cv::putText(imgColor1, "features: " + std::to_string(keys1.size()),
              cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(0, 255, 255), 1);
  cv::putText(imgColor2, "features: " + std::to_string(keys2.size()),
              cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
              cv::Scalar(0, 255, 255), 1);
  cv::imshow("image1", imgColor1);
  cv::imshow("image2", imgColor2);
  cv::waitKey(
      1); // Set to 1 ms so the windows draw without blocking code execution

  // Match descriptors using variable object instance 'lighterglue'
  std::cout << "matching ...\n";
  std::vector<cv::DMatch> matches;
  lighterglue.Match(keys1, descs1, img1_size, keys2, descs2, img2_size,
                    matches);

  // Render matching results
  cv::Mat imgMatches;
  cv::drawMatches(imgColor1, keys1, imgColor2, keys2, matches, imgMatches);
  cv::imshow("matches", imgMatches);
  cv::waitKey(0);

  return 0;
}
