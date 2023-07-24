#include "mpc_tracking/mpc.h"

Mpc::Mpc() {
    mpc_window_ = 30;
    dt_ = 0.1;
    x0_ = Eigen::Vector2d::Zero();
    x_ref_ << 3, 3;
    setDynamicsMatrices(a_, b_);
    setInequalityConstraints(x_max_, x_min_, u_max_, u_min_);
    setWeightMatrices(Q_, R_);

    //cast the MPC problem as QP problem
    castMPCToQPHessian(Q_, R_, mpc_window_, hessian_);
    castMPCToQPGradient(Q_, x_ref_, mpc_window_, gradient_);
    castMPCToQPConstraintMatrix(a_, b_, mpc_window_, linear_matrix_);
    castMPCToQPConstraintVectors(x_max_, x_min_, u_max_, u_min_, x0_, mpc_window_, lower_bound_, upper_bound_);
    
}
Mpc::~Mpc() {}

void Mpc::solveMpc(Eigen::Vector2d x0) {
    
    updateConstraintVectors(x0, lower_bound_, upper_bound_);
    // instantiate the solver
    OsqpEigen::Solver solver;

    //settings
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);

    //set the initial data of the QP solver
    solver.data()->setNumberOfVariables(2 * (mpc_window_ + 1) + 2 * mpc_window_);
    solver.data()->setNumberOfConstraints(2 * 2 * (mpc_window_ + 1) + 2 * mpc_window_);
    if (!solver.data()->setHessianMatrix(hessian_)) return;
    if (!solver.data()->setGradient(gradient_)) return;
    if (!solver.data()->setLinearConstraintsMatrix(linear_matrix_)) return;
    if (!solver.data()->setLowerBound(lower_bound_)) return;
    if (!solver.data()->setUpperBound(upper_bound_)) return;

    // instantiate the solver
    if (!solver.initSolver()) return;

    //controller input and QPSolution vector

    // solve the QP problem
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return;

    // get QP solution
    QP_solution_ = solver.getSolution();
    //cout << "QP_solution_ :" << QP_solution_.transpose()  << endl; 
    
}

void Mpc::solveMpc(Eigen::Vector2d x0, Eigen::Matrix2Xd &x_ref) {
    // update x0
    updateConstraintVectors(x0, lower_bound_, upper_bound_);

    // update xref
    castMPCToQPGradientVarXref(Q_, x_ref, mpc_window_, gradient_);

    // instantiate the solver
    OsqpEigen::Solver solver;

    //settings
    solver.settings()->setVerbosity(false);
    solver.settings()->setWarmStart(true);

    //set the initial data of the QP solver
    solver.data()->setNumberOfVariables(2 * (mpc_window_ + 1) + 2 * mpc_window_);
    solver.data()->setNumberOfConstraints(2 * 2 * (mpc_window_ + 1) + 2 * mpc_window_);
    if (!solver.data()->setHessianMatrix(hessian_)) return;
    if (!solver.data()->setGradient(gradient_)) return;
    if (!solver.data()->setLinearConstraintsMatrix(linear_matrix_)) return;
    if (!solver.data()->setLowerBound(lower_bound_)) return;
    if (!solver.data()->setUpperBound(upper_bound_)) return;

    // instantiate the solver
    if (!solver.initSolver()) return;

    //controller input and QPSolution vector

    // solve the QP problem
    if (solver.solveProblem() != OsqpEigen::ErrorExitFlag::NoError) return;

    // get QP solution
    QP_solution_ = solver.getSolution();
    //cout << "QP_solution_ :" << QP_solution_.transpose()  << endl; 
 
}

Eigen::Vector2d Mpc::getControlCmd() {
    Eigen::Vector2d cmd;
    cmd = QP_solution_.block(2 * (mpc_window_ + 1), 0, 2, 1);
    return cmd;
}

vector<double> Mpc::getPredictState() {
    vector<double> predict_states;
    for (int i = 0; i < mpc_window_; ++i) {
        predict_states.push_back(QP_solution_(2 * i, 0));
        predict_states.push_back(QP_solution_(2 * i + 1, 0));
    }
    return predict_states;
}

void Mpc::setDynamicsMatrices(Eigen::Matrix2d &a, Eigen::Matrix2d &b) {
    a = Eigen::Matrix2d::Identity();
    b << dt_, 0.0, 0.0, dt_;
}

void Mpc::setInequalityConstraints(Eigen::Vector2d &x_max, Eigen::Vector2d &x_min,
                                    Eigen::Vector2d &u_max, Eigen::Vector2d &u_min) {
    // input inequality constraints
    u_min << -3, -3;
    u_max << 3.0, 3.0;

    // state inequality constraints  
    x_min << -OsqpEigen::INFTY, -OsqpEigen::INFTY;
    x_max << OsqpEigen::INFTY, OsqpEigen::INFTY;                                
}

void Mpc::setWeightMatrices(Eigen::DiagonalMatrix<double, 2> &Q, Eigen::DiagonalMatrix<double, 2> R) {
    Q.diagonal() << 0.01 ,0.01 ;
    R.diagonal() << 0.1, 0.1;
}

void Mpc::castMPCToQPHessian(const Eigen::DiagonalMatrix<double, 2> &Q, const Eigen::DiagonalMatrix<double, 2> R,
                            int mpc_window, Eigen::SparseMatrix<double> &hessian_matrix) {

    hessian_matrix.resize(2 * (mpc_window + 1) + 2 * mpc_window, 2 * (mpc_window + 1) + 2 * mpc_window);

    //populate hessian matrix
    for (int i = 0; i < 2 * (mpc_window + 1) + 2 * mpc_window; ++i) {
        if (i < 2 * (mpc_window + 1)) { //Q Q_N
            int posQ = i % 2;
            float value = Q.diagonal()[posQ];
            if (value != 0) {
                hessian_matrix.insert(i, i) = value;
            }
        } else { //R
            int posR = i % 2;
            float value = R.diagonal()[posR];
            if (value != 0) {
                hessian_matrix.insert(i, i) = value;
            }
        }
    }
}

void Mpc::castMPCToQPGradient(const Eigen::DiagonalMatrix<double, 2> &Q, const Eigen::Vector2d &x_ref,
                            int mpc_window, Eigen::VectorXd &gradient) {
    // populate the gradient vector
    gradient = Eigen::VectorXd::Zero(2 * (mpc_window + 1) + 2 * mpc_window, 1);
    Eigen::Vector2d Qx_ref = Q * (-x_ref);
    for (int i = 0; i < 2 * (mpc_window + 1); ++i) {
        int posQ = i % 2;
        float value = Qx_ref(posQ, 0);
        gradient(i, 0) = value;
    }                            
}

void Mpc::castMPCToQPGradientVarXref(const Eigen::DiagonalMatrix<double, 2> &Q, const Eigen::Matrix2Xd &x_ref,
                            int mpc_window, Eigen::VectorXd &gradient) {
    // populate the gradient vector
    gradient = Eigen::VectorXd::Zero(2 * (mpc_window + 1) + 2 * mpc_window, 1);

    for (int i = 0; i < mpc_window + 1; ++i) {
        Eigen::Vector2d Qx_ref = Q * (-x_ref.col(i));
        gradient.block(2 * i, 0, 2, 1) = Qx_ref;
    }                            
}

void Mpc::castMPCToQPConstraintMatrix(const Eigen::Matrix2d &dynamic_matrix, const Eigen::Matrix2d &control_matrix,
                                    int mpc_window, Eigen::SparseMatrix<double> &constraint_matrix) {
    constraint_matrix.resize(2 * (mpc_window + 1) + 2 * (mpc_window + 1) + 2 * mpc_window, 2 * (mpc_window + 1) + 2 * mpc_window);
    //populate linear constraint matrix
    for (int i = 0; i < 2 * (mpc_window + 1); ++i) {
        constraint_matrix.insert(i, i) = -1;
    }                                    

    for (int i = 0; i < mpc_window; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                float value = dynamic_matrix(j, k);
                if (value != 0) {
                    constraint_matrix.insert(2 * (i + 1) + j, 2 * i + k) = value;
                }
            }
        }
    }

    for (int i = 0; i < mpc_window; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 2; ++k) {
                float value = control_matrix(j, k);
                if (value != 0) {
                    constraint_matrix.insert(2 * (i + 1) + j, 2 * i + k + 2 * (mpc_window + 1)) = value;
                }
            }
        }
    }

    for (int i = 0; i < 2 * (mpc_window + 1) + 2 * mpc_window; ++i) {
        constraint_matrix.insert(2 * (mpc_window + 1) + i, i) = 1;
    }
}
void Mpc::castMPCToQPConstraintVectors(const Eigen::Vector2d &x_max, const Eigen::Vector2d &x_min,
                                        const Eigen::Vector2d &u_max, const Eigen::Vector2d &u_min,
                                        const Eigen::Vector2d &x0, int mpc_window,
                                        Eigen::VectorXd &lower_bound, Eigen::VectorXd &upper_bound) {
    // evaluate the lower and the upper inequality vectors
    Eigen::VectorXd lower_inequality = Eigen::MatrixXd::Zero(2 * (mpc_window + 1) + 2 * mpc_window, 1);
    Eigen::VectorXd upper_inequality = Eigen::MatrixXd::Zero(2 * (mpc_window + 1) + 2 * mpc_window, 1); 

    for (int i = 0; i < mpc_window + 1; ++i) {
        lower_inequality.block(2 * i, 0, 2, 1) = x_min;
        upper_inequality.block(2 * i, 0, 2, 1) = x_max;
    }                                       
    for (int i = 0; i < mpc_window; ++i) {
        lower_inequality.block(2 * i + 2 * (mpc_window + 1), 0, 2, 1) = u_min;
        upper_inequality.block(2 * i + 2 * (mpc_window + 1), 0, 2, 1) = u_max;
    }
    // evaluate the lower and the upper equality vectors
    Eigen::VectorXd lower_equality = Eigen::VectorXd::Zero(2 * (mpc_window + 1), 1);
    Eigen::VectorXd upper_equality;
    lower_equality.block(0, 0, 2, 1) = -x0;
    upper_equality = lower_equality;
    lower_equality = lower_equality; //?

    //merge inequality and equality vectors
    lower_bound = Eigen::VectorXd::Zero(2 * 2 * (mpc_window + 1) + 2 * mpc_window, 1);
    upper_bound = Eigen::VectorXd::Zero(2 * 2 * (mpc_window + 1) + 2 * mpc_window, 1);

    lower_bound << lower_equality, lower_inequality;
    upper_bound << upper_equality, upper_inequality;
     
}
void Mpc::updateConstraintVectors(const Eigen::Vector2d &x0, Eigen::VectorXd &lower_bound, Eigen::VectorXd &upper_bound) {
    lower_bound.block(0, 0, 2, 1) = -x0;
    upper_bound.block(0, 0, 2, 1) = -x0;
}