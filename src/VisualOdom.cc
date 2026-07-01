#include <Eigen/Core>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/src/Core/util/Constants.h>
#include <eigen3/Eigen/SVD>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/types.hpp>

class VisualOdom {
public:
  void get_essential_matrix(std::vector<cv::DMatch> &matches,
                            std::vector<cv::KeyPoint> &kpts0,
                            std::vector<cv::KeyPoint> &kpts1) {

    Eigen::Matrix<float, Eigen::Dynamic, 9> A(matches.size());
    Eigen::DiagonalMatrix<float, Eigen::Dynamic> w(matches.size());
    Eigen::RowVector3f train, query;

    for (int i = 0; i < matches.size(); i++) {
      train = {kpts0[matches[i].trainIdx].pt.x, kpts0[matches[i].queryIdx].pt.y,
               1};
      query = {kpts1[matches[i].queryIdx].pt.x, kpts1[matches[i].queryIdx].pt.y,
               1};
      A.row(i) = Eigen::kroneckerProduct(train, query);
      w.diagonal()[i] = 1.0 - matches[i].distance;
    }

    Eigen::Matrix<float, Eigen::Dynamic, 9> wA = w * A;

    Eigen::BDCSVD svd(wA, Eigen::ComputeThinV);

    Eigen::Vector<float, 9> eta = svd.matrixV().col(8);

    Eigen::Matrix3f E =
        Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(eta.data());

    Eigen::JacobiSVD E_svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::Matrix3f V_E_svd = E_svd.matrixV();
    Eigen::Vector3f sigma(1, 1, 0);
    Eigen::Matrix3f E_approx =
        E_svd.matrixU() * sigma.asDiagonal() * V_E_svd.transpose();

    Eigen::Vector3f t = V_E_svd.col(2);

    // construct skew matrix of t
    Eigen::Matrix3f t_x = Eigen::Matrix3f::Zero();
    t_x(0, 1) = -t(2); // -tz
    t_x(0, 2) = t(1);  //  ty
    t_x(1, 0) = t(2);  //  tz
    t_x(1, 2) = -t(0); // -tx
    t_x(2, 0) = -t(1); // -ty
    t_x(2, 1) = t(0);  //  tx

    Eigen::JacobiSVD t_x_svd(t_x, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3f t_x_U = t_x_svd.matrixU();
    Eigen::Matrix3f t_x_V = t_x_svd.matrixV();

    Eigen::Matrix3f W1 = (t_x_U.col(0) * t_x_V.col(0).transpose()) +
                         (t_x_U.col(1) * t_x_V.col(1).transpose()) + 
                         (t_x_U.col(2) * t_x_V.col(2).transpose());
    Eigen::Matrix3f W2 = (t_x_U.col(0) * t_x_V.col(0).transpose()) +
                         (t_x_U.col(1) * t_x_V.col(1).transpose()) -
                         (t_x_U.col(2) * t_x_V.col(2).transpose());

    Eigen::Matrix3f R1 = W1 * W1.determinant(); 
    Eigen::Matrix3f R2 = W2 * W2.determinant(); 
  }
};
