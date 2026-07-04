# pragma once 

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

class VisualOdom {

private:
  void _getProjectionMatrix(Eigen::Matrix<float, 3, 4> &P);

  void _getBearingVector(const std::vector<Eigen::RowVector3f> &x,
                         std::vector<Eigen::RowVector3f> &f);

  bool _checkPose(const Eigen::Matrix3f &R_, const Eigen::Vector3f &t_,
                  const std::vector<Eigen::RowVector3f> &f0,
                  const std::vector<Eigen::RowVector3f> &f1);

public:
  const Eigen::Matrix3f K; // camera intrinsic matrix
  Eigen::Matrix3f R;       // camera rotation matrix
  Eigen::Vector3f t;       // camera translation matrix

  VisualOdom(float fx, float fy, float Cx, float Cy, float s);

  void getCameraT(std::vector<cv::DMatch> &matches,
                  std::vector<cv::KeyPoint> &kpts0,
                  std::vector<cv::KeyPoint> &kpts1);
};
