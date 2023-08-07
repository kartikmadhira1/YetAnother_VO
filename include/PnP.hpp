#ifndef PNP_H
#define PNP_H

#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"

#include <sophus/se3.hpp>
#include <iostream>


using namespace std;



// Define the Vertex and Edge classes for the Pnp problem.


// // Vertex
// class PnPVertex : public g2o::BaseVertex<6, Sophus::SE3d>
// {
//     public:
//         EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//         virtual bool read(istream& in) override {return true;}
//         virtual bool write(ostream& out) const override{return true;}

//         virtual void setToOriginImpl() override {
//             _estimate = Sophus::SE3d();
//         }

//         // update the optimization variable => Pose, with a delta in the vector space.
//         // for small updates, P(t+1) = exp(update^)*P(t)
//         virtual void oplusImpl(const double* update) override {
//             //se(3) is a 6D vector space
//             Eigen::Matrix<double, 6, 1> update_;

//             update_ << update[0], update[1], update[2], update[3], update[4], update[5];
//             _estimate = Sophus::SE3d::exp(update_) * _estimate;
//         }

// };


class PnPVertex : public g2o::BaseVertex<6, Sophus::SE3d> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override { _estimate = Sophus::SE3d(); }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4],
            update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }
};

// class PnPEdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, PnPVertex> {
//     public:
//         EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//         PnPEdgeProjection(const Eigen::Vector3d& point, const Eigen::Matrix3d &K) {
//             _point = point;
//             _K = K;
//         }

//         virtual void computeError() override {
//             // Get the pose estimate
//             const PnPVertex* v = static_cast<const PnPVertex*>(this->_vertices[0]);
//             Sophus::SE3d T = v->estimate();

//             // get the pixel position
//             Eigen::Vector3d pCam = _K*(T*_point);
//             // Eigen::Vector2d pCamPixel(pCam[0]/pCam[2], pCam[1]/pCam[2]);
//             pCam = pCam / pCam[2];
//             // compute the error between actual and measured projection position
//             _error = _measurement - pCam.head<2>();
            // LOG(INFO) << "error: " << _error;
            // LOG(INFO) << "measurement" << _measurement;
            // LOG(INFO) << "pcampixel" << pCam.head<2>();
//             // // std::cout << "error: " << _error << std::endl;
//             // // std::cout << pCamPixel << std::endl;
//             // // std::cout << _measurement << std::endl;
//         }

//         // Jacobian calculation de/dT
//         virtual void linearizeOplus() override {

//             const PnPVertex* v = static_cast<const PnPVertex*>(this->_vertices[0]);
//             Sophus::SE3d T = v->estimate();
            
//             // Note that this 3d point will be in the camera frame! Refer to the notes for the derivation.
//             // The Jacobian calculation is done on points in the camera frame -> X`, Y`, Z`
//             Eigen::Vector3d pos_cam = T * _point;
//             double fx = _K(0, 0);
//             double fy = _K(1, 1);
//             double X = pos_cam[0];
//             double Y = pos_cam[1];
//             double Z = pos_cam[2];
//             double Zinv = 1.0 / (Z + 1e-18);
//             double Zinv2 = Zinv * Zinv;
//             double Z2 = Z * Z;
//             _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
//             -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
//             fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
//             -fy * X * Zinv;

//         }

//         virtual bool read(istream &in) override {return true; }
//         virtual bool write(ostream &out) const override {return true; }
//     private:
//         Eigen::Vector3d _point;
//         Eigen::Matrix3d _K;
// };



class PnPEdgeProjection : public g2o::BaseUnaryEdge<2, Vec2, PnPVertex> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    PnPEdgeProjection(const Vec3 &pos, const Eigen::Matrix3d &K)
        : _pos3d(pos), _K(K) {}

    virtual void computeError() override {
        const PnPVertex *v = static_cast<PnPVertex *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        // convert 3d point to homogenoi
        Vec3 pos_pixel = _K * (T * _pos3d);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
      
        
    }

    virtual void linearizeOplus() override {
        const PnPVertex *v = static_cast<PnPVertex *>(_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Vec3 pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Zinv = 1.0 / (Z + 1e-18);
        double Zinv2 = Zinv * Zinv;
        _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
            -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
            fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
            -fy * X * Zinv;
    }

    virtual bool read(std::istream &in) override { return true; }

    virtual bool write(std::ostream &out) const override { return true; }

   private:
    Vec3 _pos3d;
    Eigen::Matrix3d _K;
};



#endif