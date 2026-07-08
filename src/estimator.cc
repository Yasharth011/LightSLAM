#include "estimator.h"

struct Gyro_bias_init {
    Gyro_bias_init(Eigen::Quaternion<T>& frame_prev, Eigen::Quaternion<T>& frame_next_inv, Eigen::Quaternion<T>& rot_preint)
        :frame_prev(frame_prev), frame_next(frame_next_inv), rot_preint(rot_preint);

    template <typename T>
    bool operator()(const T* delta_bg, T* residual) const {
        Eigen::Quaternion<T> q_err;
        q_err = frame_next_inv*frame_prev*rot_preint;
        residual[0] = q_err.w;
        residual[1] = q_err.x;
        residual[2] = q_err.y;
        residual[3] = q_err.z;

        Eigen::Map<const Eigen::Matrix<T, 3, 1>> d_bg(delta_bg);
        Eigen::Quaternion<T> 
        return true;
    }
};


void Estimator::gyro_bias_cal (Eigen::Matrix3d& frame_prev, Eigen::Matrix3d& frame_next, Eigen::Matrix3d& rot_preint ){
     Eigen::Quaternionf q_prev(frame_prev);
     Eigen::Quaternionf q_next(frame_next);
     Eigen::Quaternionf q_rot_preint(rot_preint);

     Eigen::Quaternionf q_next_inv = q_next.inverse()

    }