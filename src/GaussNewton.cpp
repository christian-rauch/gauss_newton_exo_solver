#include <GaussNewton/GaussNewton.hpp>

REGISTER_MOTIONSOLVER_TYPE("GNsolver", exotica::GaussNewton)

namespace exotica
{
GaussNewton::GaussNewton() : iterations_(-1) { }

GaussNewton::~GaussNewton() { }

void GaussNewton::Instantiate(GNsolverInitializer& init) { parameters_ = init; }

void GaussNewton::specifyProblem(PlanningProblem_ptr pointer) {
    if (pointer->type() != "exotica::UnconstrainedEndPoseProblem") {
        throw_named("This GaussNewton can't solve problem of type '" << pointer->type() << "'!");
    }

    MotionSolver::specifyProblem(pointer);

    prob_ = std::static_pointer_cast<UnconstrainedEndPoseProblem>(pointer);
}

void GaussNewton::Solve(Eigen::MatrixXd& solution) {
    prob_->resetCostEvolution(getNumberOfMaxIterations() + 1);

    Timer timer;

    if (!prob_) throw_named("Solver has not been initialized!");
    const Eigen::VectorXd q0 = prob_->applyStartState();

    if (prob_->N != q0.rows()) throw_named("Wrong size q0 size=" << q0.rows() << ", required size=" << prob_->N);

    solution.resize(1, prob_->N);

    Eigen::VectorXd q = q0;
    double error = INFINITY;
    int i;
    for (i = 0; i < getNumberOfMaxIterations(); i++)
    {
        prob_->Update(q);
        Eigen::VectorXd yd = prob_->Cost.S * prob_->Cost.ydiff;

        std::cout << "yd: " << std::endl << std::setprecision(3) << yd.transpose() << std::endl;

        error = prob_->getScalarCost();

        std::cout << "err: " << std::endl << error << ", " << yd.cwiseAbs().sum() << std::endl;

        prob_->setCostEvolution(i, error);

//        std::cout << "S: " << std::endl << std::setprecision(3) << prob_->Cost.S << std::endl;
        std::cout << "J: " << std::endl << std::setprecision(3) << prob_->Cost.J << std::endl;
        //Eigen::MatrixXd Jinv = PseudoInverse(prob_->Cost.S * prob_->Cost.J);
        Eigen::MatrixXd Jinv = (prob_->Cost.S*prob_->Cost.J).completeOrthogonalDecomposition().pseudoInverse();
        std::cout << "Jinv: " << std::endl << std::setprecision(3) << Jinv << std::endl;
        Eigen::VectorXd qd = Jinv * yd;
        std::cout << "qd: " << std::endl << std::setprecision(3) << qd.transpose() << std::endl;

        ScaleToStepSize(qd);

        std::cout << "qds: " << std::endl << std::setprecision(3) << qd.transpose() << std::endl;

        if (parameters_.Alpha.size() == 1)
        {
            q -= qd * parameters_.Alpha[0];
        }
        else
        {
            q -= qd.cwiseProduct(parameters_.Alpha);
        }

        std::cout << "q: " << std::endl << std::setprecision(3) << q.transpose() << std::endl;

        if (qd.norm() < parameters_.Convergence)
        {
            if (debug_) HIGHLIGHT_NAMED("IKsolver", "Reached convergence (" << qd.norm() << " < " << parameters_.Convergence << ")");
            break;
        }
    }
    iterations_ = i + 1;

    solution.row(0) = q;

    planning_time_ = timer.getDuration();
}

void GaussNewton::ScaleToStepSize(Eigen::VectorXdRef xd)
{
    double max_vel = xd.cwiseAbs().maxCoeff();
    if (max_vel > parameters_.MaxStep)
    {
        xd = xd * parameters_.MaxStep / max_vel;
    }
}

}
