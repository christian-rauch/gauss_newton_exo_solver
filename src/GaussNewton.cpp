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

    lambda = parameters_.Damping;   // initial damping

//    std::cout << "JN " << prob_->JN << std::endl;
//    std::cout << "PhiN " << prob_->PhiN << std::endl;

    const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(prob_->Cost.J.cols(), prob_->Cost.J.cols());

    Eigen::VectorXd q = q0;
    double error = std::numeric_limits<double>::infinity();
    Eigen::VectorXd qd;
    int i;
    for (i = 0; i < getNumberOfMaxIterations(); i++)
    {
        prob_->Update(q);
        Eigen::VectorXd yd = prob_->Cost.S * prob_->Cost.ydiff;

        std::cout << "yd: " << std::endl << std::setprecision(3) << yd.transpose() << std::endl;

        // weightes sum of squares
        error = prob_->getScalarCost();

        std::cout << "err: " << std::endl << error << ", " << yd.cwiseAbs().sum() << std::endl;

        prob_->setCostEvolution(i, error);

        const double mse = error/yd.size();
        std::cout << "mse: " << mse << std::endl;

//        std::cout << "S: " << std::endl << std::setprecision(3) << prob_->Cost.S << std::endl;
//        std::cout << "J: " << std::endl << std::setprecision(3) << prob_->Cost.J << std::endl;

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
        std::cout << "damping: " << lambda << std::endl;

        // via inverse
//        Eigen::MatrixXd Jinv = (prob_->Cost.S*prob_->Cost.J).completeOrthogonalDecomposition().pseudoInverse();
//        std::cout << "Jinv: " << std::endl << std::setprecision(3) << Jinv << std::endl;
//        qd = Jinv * yd;

        // via solve
        qd = (prob_->Cost.J.transpose()*prob_->Cost.J + lambda*I).ldlt().solve(prob_->Cost.J.transpose()*prob_->Cost.ydiff);

//        const double relative_error = (prob_->Cost.J.transpose()*prob_->Cost.J * qd - prob_->Cost.J.transpose()*prob_->Cost.ydiff).norm() / (prob_->Cost.J.transpose()*prob_->Cost.ydiff).norm(); // norm() is L2 norm
//        std::cout << "relative_error: " << relative_error << std::endl;

        std::cout << "qd: " << std::endl << std::setprecision(3) << qd.transpose() << std::endl;

//        ScaleToStepSize(qd);
//        std::cout << "qds: " << std::endl << std::setprecision(3) << qd.transpose() << std::endl;

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
