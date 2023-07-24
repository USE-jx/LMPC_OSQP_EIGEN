#ifndef MPC_H
#define MPC_H

#include <vector>
#include <Eigen/Dense>
#include "tf2/utils.h"
#include <casadi/casadi.hpp>
#include <iostream>
#include <chrono>
#include "OsqpEigen/OsqpEigen.h"

using namespace std;
using namespace casadi;

class Mpc
{
private:
    //preview window
    int mpc_window_;
    double dt_;

    //dynamics matrices
    Eigen::Matrix2d a_, b_;

    //constraints vector
    Eigen::Vector2d x_max_, x_min_, u_max_, u_min_;

    //weight matrices
    Eigen::DiagonalMatrix<double, 2> Q_, R_;

    //the initial and the reference state
    Eigen::Vector2d x0_, x_ref_;

    //QP problem matrices and vectors
    Eigen::SparseMatrix<double> hessian_;
    Eigen::VectorXd gradient_;
    Eigen::SparseMatrix<double> linear_matrix_;
    Eigen::VectorXd lower_bound_, upper_bound_; 

    // QP solution
    Eigen::VectorXd QP_solution_;

    
 

public:
    Mpc();
    ~Mpc();


    void setDynamicsMatrices(Eigen::Matrix2d &a, Eigen::Matrix2d &b);
    void setInequalityConstraints(Eigen::Vector2d &x_max, Eigen::Vector2d &x_min,
                                    Eigen::Vector2d &u_max, Eigen::Vector2d &u_min);
    void setWeightMatrices(Eigen::DiagonalMatrix<double, 2> &Q, Eigen::DiagonalMatrix<double, 2> R);
    void castMPCToQPHessian(const Eigen::DiagonalMatrix<double, 2> &Q, const Eigen::DiagonalMatrix<double, 2> R,
                            int mpc_window, Eigen::SparseMatrix<double> &hessian_matrix);
    void castMPCToQPGradient(const Eigen::DiagonalMatrix<double, 2> &Q, const Eigen::Vector2d &x_ref,
                            int mpc_window, Eigen::VectorXd &gradient);
    
    void castMPCToQPGradientVarXref(const Eigen::DiagonalMatrix<double, 2> &Q, const Eigen::Matrix2Xd &x_ref,
                            int mpc_window, Eigen::VectorXd &gradient);
    void castMPCToQPConstraintMatrix(const Eigen::Matrix2d &dynamic_atrix, const Eigen::Matrix2d &control_matrix,
                                    int mpc_window, Eigen::SparseMatrix<double> &constraint_matrix);
    void castMPCToQPConstraintVectors(const Eigen::Vector2d &x_max, const Eigen::Vector2d &x_min,
                                        const Eigen::Vector2d &u_max, const Eigen::Vector2d &u_min,
                                        const Eigen::Vector2d &x0, int mpc_window,
                                        Eigen::VectorXd &lower_bound, Eigen::VectorXd &upper_bound);
    void updateConstraintVectors(const Eigen::Vector2d &x0, Eigen::VectorXd &lower_bound, Eigen::VectorXd &upper_bound);
    
    void solveMpc(Eigen::Vector2d x0);
    void solveMpc(Eigen::Vector2d x0, Eigen::Matrix2Xd &x_ref);
    Eigen::Vector2d getControlCmd();
    vector<double> getPredictState();
    
};

#endif