#include <eigen3/Eigen/SVD>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <opencv2/calib3d.hpp>
#include <opencv2/ml.hpp>

#include <VisualOdom.h>

// clang-format off
VisualOdom::VisualOdom(float fx, float fy, float Cx, float Cy, float s):
      K((Eigen::Matrix3f() <<  fx,   s,   Cx, 
			      0.f,  fy,   Cy, 
			      0.f, 0.f,  1.f).finished()) 
  {};
// clang-format on

void VisualOdom::_getProjectionMatrix(Eigen::Matrix<float, 3, 4> &P) {
  Eigen::Matrix<float, 3, 4> Rt;
  Rt.block<3, 3>(0, 0) = R;
  Rt.block<3, 1>(0, 3) = t;
  P = K * Rt;
}

void VisualOdom::_getBearingVector(const std::vector<Eigen::RowVector3f> &x,
                                   std::vector<Eigen::RowVector3f> &f) {
  auto K_ = K.inverse();
  for (int i = 0; i < x.size(); i++) {
    f[i] = (K * x[i].transpose()).transpose();
  }
}

bool VisualOdom::_checkPose(const Eigen::Matrix3f &R_,
                            const Eigen::Vector3f &t_,
                            const std::vector<Eigen::RowVector3f> &f0,
                            const std::vector<Eigen::RowVector3f> &f1) {

  bool flag = false;

  // construct skew matrix of t_
  Eigen::Matrix3f t_x;
  t_x << 0, -t_[2], t_[1], t_[2], 0, -t_[0], -t_[1], t_[0], 0;

  // construct Essential Matrix E
  Eigen::Matrix3f E = t_x * R_;

  Eigen::Vector3f q = R_.transpose() * t_;

  for (int i = 0; i < f0.size(); i++) {

    // rotation ambiguity
    if (float(((E * f1[i].transpose()).dot(t_x * f0[i].transpose()))) <= 0.f) {
      flag = false;
      break;
    }

    // translation ambiguity
    else if (float((f0[i].dot(t_)) - (f1[i].dot(q))) < 0.f) {
      flag = false;
      break;
    }

    else
      flag = true;
  }
  return flag;
}


void VisualOdom::getCameraT(std::vector<cv::DMatch> &matches,
                std::vector<cv::KeyPoint> &kpts0,
                std::vector<cv::KeyPoint> &kpts1) {

  // Compute Kronecker product of diag(w)*A
  Eigen::Matrix<float, Eigen::Dynamic, 9> A;
  Eigen::DiagonalMatrix<float, Eigen::Dynamic> w;
  Eigen::RowVector3f t, q;
  std::vector<Eigen::RowVector3f> train, query;

  for (int i = 0; i < matches.size(); i++) {

    t = {kpts0[matches[i].trainIdx].pt.x, kpts0[matches[i].trainIdx].pt.y, 1};
    q = {kpts1[matches[i].queryIdx].pt.x, kpts1[matches[i].queryIdx].pt.y, 1};

    train.push_back(t);
    query.push_back(q);

    A.row(i) = Eigen::kroneckerProduct(t, q);

    w.diagonal()[i] = 1.0 - matches[i].distance;
  }

  // Compute Essential Matrix E
  Eigen::Matrix<float, Eigen::Dynamic, 9> wA = w * A;

  Eigen::BDCSVD svd(wA, Eigen::ComputeThinV);

  Eigen::Vector<float, 9> eta = svd.matrixV().col(8);

  Eigen::Matrix3f E =
      Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(eta.data());

  Eigen::JacobiSVD E_svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

  Eigen::Matrix3f V = E_svd.matrixV();
  Eigen::Matrix3f U = E_svd.matrixU();

  Eigen::Vector3f sigma(1, 1, 0);
  Eigen::Matrix3f E_approx = U * sigma.asDiagonal() * V.transpose();

  // Decompose E -> R and t
  Eigen::Matrix3f W;
  // clang-format off
    W << 0, -1, 0, 
      	 1,  0, 0, 
    	 0,  0, 1;
  // clang-format on

  Eigen::Matrix3f R1 = U * W * V.transpose();
  Eigen::Matrix3f R2 = U * W.transpose() * V.transpose();
  Eigen::Vector3f t1 = U.col(2);
  Eigen::Vector3f t2 = -1 * U.col(2);
  const std::pair<Eigen::Matrix3f, Eigen::Vector3f> candidates[4] = {
      {R1, t1}, {R1, t2}, {R2, t1}, {R2, t2}};

  // get bearing vector of images
  std::vector<Eigen::RowVector3f> f0, f1;
  _getBearingVector(train, f0);
  _getBearingVector(query, f1);

  // check correct pose using C2P chierality check
  for (const auto &[R_, t_] : candidates) {
    if (_checkPose(R_, t_, f0, f1)) {
      R = R_;
      t = t_;
      return;
    }
  }
}
