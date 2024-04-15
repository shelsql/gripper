
#include <iostream>
#include <ceres/ceres.h>
 
// 定义代价函数
struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
        residual[0] = T(10.0) - x[0];
        return true;
    }
};
 
int main() {
    double initial_x = 5.0;
    double x = initial_x;
 
    ceres::Problem problem;
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(cost_function, nullptr, &x);
 
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
 
    std::cout << summary.BriefReport() << "\n";
    std::cout << "Initial x: " << initial_x << "\n";
    std::cout << "Final x: " << x << "\n";
 
    return 0;
}