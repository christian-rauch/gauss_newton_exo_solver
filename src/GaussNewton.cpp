#include <exotica_core/problems/unconstrained_end_pose_problem.h>
#include <exotica_core/motion_solver.h>
#include <gn_solver/GNsolver_initializer.h>


namespace exotica {

class GaussNewton : public MotionSolver, public Instantiable<GNsolverInitializer> {
public:
    virtual void Solve(Eigen::MatrixXd& solution) override;

    virtual void SpecifyProblem(PlanningProblemPtr pointer) override;

private:
    UnconstrainedEndPoseProblemPtr prob_;  // Shared pointer to the planning problem.

    double lambda = 0;  // damping factor
};

REGISTER_MOTIONSOLVER_TYPE("GNsolver", exotica::GaussNewton)

void GaussNewton::SpecifyProblem(PlanningProblemPtr pointer) {
    if (pointer->type() != "exotica::UnconstrainedEndPoseProblem") {
        ThrowNamed("This GaussNewton can't solve problem of type '" << pointer->type() << "'!");
    }

    MotionSolver::SpecifyProblem(pointer);

    // generic problem
    problem_ = pointer;

    // specific problem
    prob_ = std::static_pointer_cast<UnconstrainedEndPoseProblem>(pointer);

    // check dimension of alpha
    if (parameters_.Alpha.size() > 1 && parameters_.Alpha.size() != this->problem_->N) {
        ThrowNamed("Wrong alpha dimension: alpha(" << parameters_.Alpha.size() << ") != states(" << this->problem_->N << ")")
    }
}

void GaussNewton::Solve(Eigen::MatrixXd& solution) {
    prob_->ResetCostEvolution(GetNumberOfMaxIterations() + 1);

    Timer timer;

    if (!prob_) ThrowNamed("Solver has not been initialized!");

    const Eigen::VectorXd q0 = prob_->ApplyStartState();

    if (prob_->N != q0.rows()) ThrowNamed("Wrong size q0 size=" << q0.rows() << ", required size=" << prob_->N);

    solution.resize(1, prob_->N);

    lambda = parameters_.Damping;   // initial damping

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(prob_->cost.jacobian.cols(), prob_->cost.jacobian.cols());
    Eigen::MatrixXd J;

    Eigen::VectorXd q = q0;
    double error = std::numeric_limits<double>::infinity();
    double error_prev = std::numeric_limits<double>::infinity();
    Eigen::VectorXd yd;
    Eigen::VectorXd qd;
    for(size_t i = 0; i < GetNumberOfMaxIterations(); i++) {
        prob_->Update(q);

        yd = prob_->cost.S * prob_->cost.ydiff;

        if(debug_) std::cout << "yd: " << std::endl << std::setprecision(3) << yd.transpose() << std::endl;

        // weighted sum of squares
        error_prev = error;
        error = prob_->GetScalarCost();

        if(debug_) std::cout << "err: " << std::endl << error << ", " << yd.cwiseAbs().sum() << std::endl;

        prob_->SetCostEvolution(i, error);

        if(debug_) std::cout << "mse: " << error/yd.size() << std::endl;

        J = prob_->cost.S*prob_->cost.jacobian;

//        std::cout << "S: " << std::endl << std::setprecision(3) << prob_->cost.S << std::endl;
        if(debug_) std::cout << "J: " << std::endl << std::setprecision(3) << J << std::endl;

        // source: https://uk.mathworks.com/help/optim/ug/least-squares-model-fitting-algorithms.html, eq. 13

        if(i>0) {
            if(error < error_prev) {
                // success, error decreased: decrease damping
                lambda = lambda / 10.0;
            }
            else {
                // failure, error increased: increase damping
                lambda = lambda * 10.0;
            }
        }

        if(debug_) std::cout << "damping: " << lambda << std::endl;

        Eigen::MatrixXd M;
        if(parameters_.ScaleProblem=="none") {
            M = I;
        }
        else if(parameters_.ScaleProblem=="Jacobian") {
            M = (J.transpose()*J).diagonal().asDiagonal();
        }
        else {
            throw std::runtime_error("no ScaleProblem of type "+parameters_.ScaleProblem);
        }

        // condition number
        // https://forum.kde.org/viewtopic.php?f=74&t=117430#p292018
        if(debug_) {
            const Eigen::MatrixXd A = J.transpose()*J + lambda*M;
//            std::cout << "A:" << std::endl << A <<std::endl;
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(A);
            double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
            std::cout << "cond: " << cond << std::endl;
        }

        qd = (J.transpose()*J + lambda*M).completeOrthogonalDecomposition().solve(J.transpose()*yd);

        if(debug_) std::cout << "solution?: " << ((J.transpose()*J + lambda*M)*qd).isApprox(J.transpose()*yd) << std::endl;

        if(debug_) std::cout << "qd: " << std::endl << std::setprecision(3) << qd.transpose() << std::endl;

        if (parameters_.Alpha.size() == 1) {
            q -= qd * parameters_.Alpha[0];
        }
        else {
            q -= qd.cwiseProduct(parameters_.Alpha);
        }

        if(debug_) std::cout << "q: " << std::endl << std::setprecision(3) << q.transpose() << std::endl;

        if (qd.norm() < parameters_.Convergence) {
            if (debug_) HIGHLIGHT_NAMED("GaussNewton", "Reached convergence (" << qd.norm() << " < " << parameters_.Convergence << ")");
            break;
        }
    }

    solution.row(0) = q;

    planning_time_ = timer.GetDuration();
}

}   // namespace exotica
