#ifndef GN_SOLVER_H_
#define GN_SOLVER_H_

#include <exotica/Exotica.h>
#include <exotica/Problems/UnconstrainedEndPoseProblem.h>
#include <gn_solver/GNsolverInitializer.h>

namespace exotica {
class GaussNewton : public MotionSolver, public Instantiable<GNsolverInitializer> {
public:
    GaussNewton();

    virtual ~GaussNewton();

    virtual void Instantiate(GNsolverInitializer& init);

    virtual void Solve(Eigen::MatrixXd& solution);

    virtual void specifyProblem(PlanningProblem_ptr pointer);

    UnconstrainedEndPoseProblem_ptr& getProblem() { return prob_; }

    int getLastIteration() { return iterations_; }

    void ScaleToStepSize(Eigen::VectorXdRef xd);

private:
    GNsolverInitializer parameters_;

    UnconstrainedEndPoseProblem_ptr prob_;  // Shared pointer to the planning problem.

    int iterations_;
};
}

#endif
