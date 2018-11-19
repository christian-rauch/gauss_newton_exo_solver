#include <exotica/Problems/UnconstrainedEndPoseProblem.h>
#include <exotica/MotionSolver.h>
#include <gn_solver/GNsolverInitializer.h>


namespace exotica {

class GaussNewton : public MotionSolver, public Instantiable<GNsolverInitializer> {
public:
    virtual void Instantiate(GNsolverInitializer& init);

    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void specifyProblem(PlanningProblem_ptr pointer);

private:
    GNsolverInitializer parameters_;

    UnconstrainedEndPoseProblem_ptr prob_;  // Shared pointer to the planning problem.

    double lambda = 0;  // damping factor

    int iterations_ = -1;
};

REGISTER_MOTIONSOLVER_TYPE("GNsolver", exotica::GaussNewton)

void GaussNewton::Instantiate(GNsolverInitializer& init) { parameters_ = init; }

void GaussNewton::specifyProblem(PlanningProblem_ptr pointer) {
    if (pointer->type() != "exotica::UnconstrainedEndPoseProblem") {
        throw_named("This GaussNewton can't solve problem of type '" << pointer->type() << "'!");
    }

    MotionSolver::specifyProblem(pointer);

    // generic problem
    problem_ = pointer;

    // specific problem
    prob_ = std::static_pointer_cast<UnconstrainedEndPoseProblem>(pointer);
}

void GaussNewton::Solve(Eigen::MatrixXd& solution) {
    prob_->resetCostEvolution(getNumberOfMaxIterations() + 1);

    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");

    const Eigen::VectorXd q0 = prob_->applyStartState();

    if (prob_->N != q0.rows()) throw_named("Wrong size q0 size=" << q0.rows() << ", required size=" << prob_->N);

    solution.resize(1, prob_->N);

    lambda = parameters_.Damping;   // initial damping

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(prob_->Cost.J.cols(), prob_->Cost.J.cols());
    Eigen::MatrixXd J;

    Eigen::VectorXd q = q0;
    double error = std::numeric_limits<double>::infinity();
    Eigen::VectorXd yd;
    Eigen::VectorXd qd;
    for(size_t i = 0; i < getNumberOfMaxIterations(); iterations_=++i) {
        prob_->Update(q);

        yd = prob_->Cost.S * prob_->Cost.ydiff;

        if(debug_) std::cout << "yd: " << std::endl << std::setprecision(3) << yd.transpose() << std::endl;

        // weighted sum of squares
        error = prob_->getScalarCost();

        if(debug_) std::cout << "err: " << std::endl << error << ", " << yd.cwiseAbs().sum() << std::endl;

        prob_->setCostEvolution(i, error);

        if(debug_) std::cout << "mse: " << error/yd.size() << std::endl;

        J = prob_->Cost.S*prob_->Cost.J;

//        std::cout << "S: " << std::endl << std::setprecision(3) << prob_->Cost.S << std::endl;
        if(debug_) std::cout << "J: " << std::endl << std::setprecision(3) << J << std::endl;

        if(i>0) {
            if( error < prob_->getCostEvolution(i-1) ) {
                // success: increase damping
                lambda = lambda * 10.0;
            }
            else {
                // failure: decrease damping
                lambda = lambda / 10.0;
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

    planning_time_ = timer.getDuration();
}

}   // namespace exotica
