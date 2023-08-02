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


// Vertex
class PnPVertex : public g2o::BaseVertex<6, Sophus::SE3d>
{
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        PnPVertex() {}
        virtual bool read(istream& in) {}
        virtual bool write(ostream& out) const {}

        virtual void setToOriginImpl() {
            _estimate = Sophus::SE3d();
        }

        // update the optimization variable => Pose, with a delta in the vector space.
        // for small updates, P(t+1) = exp(update^)*P(t)
        virtual void oplusImpl(const double* update) {
            //se(3) is a 6D vector space
            Eigen::Map<const Eigen::Matrix<double, 6, 1>> update_(update);
            _estimate = Sophus::SE3d::exp(update_) * _estimate;
        }

};

class PnPEdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, PnPVertex> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        PnPEdgeProjection(const Eigen::Vector3d& point, const Eigen::Matrix3d &K) {
            _point = point;
            _K = K;
        }

        virtual void computeError() override {
            // Get the pose estimate
            const PnPVertex* v = static_cast<const PnPVertex*>(this->_vertices[0]);
            Sophus::SE3d T = v->estimate();

            // get the pixel position
            Eigen::Vector3d pCam = _K*(T*_point);
            Eigen::Vector2d pCamPixel(pCam[0]/pCam[2], pCam[1]/pCam[2]);

            // compute the error between actual and measured projection position
            _error = pCamPixel - _measurement;
            // LOG(INFO) << "error: " << _error;
            // LOG(INFO) << "pcampixel" << pCamPixel;
            // LOG(INFO) << "measurement" << _measurement;
            // std::cout << "error: " << _error << std::endl;
            // std::cout << pCamPixel << std::endl;
            // std::cout << _measurement << std::endl;
        }

        // Jacobian calculation de/dT
        virtual void linearizeOplus() override {

            const PnPVertex* v = static_cast<const PnPVertex*>(this->_vertices[0]);
            Sophus::SE3d T = v->estimate();
            
            // Note that this 3d point will be in the camera frame! Refer to the notes for the derivation.
            // The Jacobian calculation is done on points in the camera frame -> X`, Y`, Z`
            Eigen::Vector3d pos_cam = T * _point;
            double fx = _K(0, 0);
            double fy = _K(1, 1);
            double X = pos_cam[0];
            double Y = pos_cam[1];
            double Z = pos_cam[2];
            double Zinv = 1.0 / (Z + 1e-18);
            double Zinv2 = Zinv * Zinv;
            double Z2 = Z * Z;
            _jacobianOplusXi << -fx * Zinv, 0, fx * X * Zinv2, fx * X * Y * Zinv2,
            -fx - fx * X * X * Zinv2, fx * Y * Zinv, 0, -fy * Zinv,
            fy * Y * Zinv2, fy + fy * Y * Y * Zinv2, -fy * X * Y * Zinv2,
            -fy * X * Zinv;

        }

        virtual bool read(istream &in) override {}
        virtual bool write(ostream &out) const override {}
    protected:
        Eigen::Vector3d _point;
        Eigen::Matrix3d _K;
};

#endif