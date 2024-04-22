#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

using namespace std;
using namespace chrono;
namespace py = pybind11;


// 代价函数的计算模型，每次计算一个点的重投影误差（test）
struct REPROJECTIONandKEY3D_COST {
    REPROJECTIONandKEY3D_COST(const double* match2d,  // 2
                      const double* match3d,  // 3
                      const double* camera_params,     // 4 fx,cx,fy,cy
                      const double* keypoint    // 3
                      ) :
    _match2d(match2d), _match3d(match3d), _camera_params(camera_params),_keypoint(keypoint)
    {}

    // 残差的计算
    template<typename T>                // T相当于一个数据类型，只不过不能把它写的具体
    bool operator()(                    // 重载括号运算符 类就相当于一个函数了   在这里实际是给人家内部自动求导用的
            const T *const q_pred,          // 待优化四元数及位移
            const T *const t_pred,
            T *residual) const {
        T pred_3dkey[3];
        T pred_2dkey[2];

        //搞不懂为啥非得这么写
        T match3d_tmp[3];
        for (int i=0;i<3;++i){
            match3d_tmp[i] = T(_match3d[i]);
        }

        // 计算预测的3d关键点位置

        ceres::QuaternionRotatePoint(q_pred,match3d_tmp,pred_3dkey);
        pred_3dkey[0] += t_pred[0]; pred_3dkey[1] += t_pred[1]; pred_3dkey[2] += t_pred[2];

        // 计算重投影位置
        pred_2dkey[0] = (pred_3dkey[0] / pred_3dkey[2])*T(_camera_params[0]) + T(_camera_params[1]);
        pred_2dkey[1] = (pred_3dkey[1] / pred_3dkey[2])*T(_camera_params[2]) + T(_camera_params[3]);
        // 计算距离
        residual[0] = 1.0*(pred_2dkey[0]-T(_match2d[0]));
        residual[1] = 1.0*(pred_2dkey[1]-T(_match2d[1]));
        // 用深度图计算keypoint的距离
        residual[2] = 1000.0*(pred_3dkey[0] - T(_keypoint[0]));
        residual[3] = 1000.0*(pred_3dkey[1] - T(_keypoint[1]));
        residual[4] = 1000.0*(pred_3dkey[2] - T(_keypoint[2]));

        return true;
    }

    const double* _match2d;
    const double* _match3d;
    const double* _camera_params;
    const double* _keypoint;
};

struct REPROJECTION_COST {
    REPROJECTION_COST(const double* match2d,  // 2
                              const double* match3d,  // 3
                              const double* camera_params     // 4 fx,cx,fy,cy
    ) :
            _match2d(match2d), _match3d(match3d), _camera_params(camera_params)
    {}

    // 残差的计算
    template<typename T>                // T相当于一个数据类型，只不过不能把它写的具体
    bool operator()(                    // 重载括号运算符 类就相当于一个函数了   在这里实际是给人家内部自动求导用的
            const T *const q_pred,          // 待优化四元数及位移
            const T *const t_pred,
            T *residual) const {
        T pred_3dkey[3];
        T pred_2dkey[2];

        //搞不懂为啥非得这么写
        T match3d_tmp[3];
        for (int i=0;i<3;++i){
            match3d_tmp[i] = T(_match3d[i]);
        }

        // 计算预测的3d关键点位置
        ceres::QuaternionRotatePoint(q_pred,match3d_tmp,pred_3dkey);
        pred_3dkey[0] += t_pred[0]; pred_3dkey[1] += t_pred[1]; pred_3dkey[2] += t_pred[2];

        // 计算重投影位置
        pred_2dkey[0] = (pred_3dkey[0] / pred_3dkey[2])*T(_camera_params[0]) + T(_camera_params[1]);
        pred_2dkey[1] = (pred_3dkey[1] / pred_3dkey[2])*T(_camera_params[2]) + T(_camera_params[3]);
        // 计算距离
        residual[0] = 1.0*(pred_2dkey[0]-T(_match2d[0]));
        residual[1] = 1.0*(pred_2dkey[1]-T(_match2d[1]));
        return true;
    }

    const double* _match2d;
    const double* _match3d;
    const double* _camera_params;
};

//struct FULL3D_COST {
//    FULL3D_COST(const double* fullpoint,    // 3
//                const vector<vector<double>> &gripper              //8192,3
//    ) :
//            _fullpoint(fullpoint), _gripper(gripper)
//    {}
//
//    // 残差的计算
//    template<typename T>                // T相当于一个数据类型，只不过不能把它写的具体
//    bool operator()(                    // 重载括号运算符 类就相当于一个函数了   在这里实际是给人家内部自动求导用的
//            const T *const q_pred,          // 待优化四元数及位移
//            const T *const t_pred,
//            T *residual) const {
//
//        unsigned num = _gripper.size();
//        // 计算预测的3d点位置
//        T pred_3d[num][3];
//        T dis[num];
//        auto time1 = system_clock::now();
//        for(int i=0;i<num;++i){
//            T gripper_tmp[3];
//            for(int j=0;j<3;++j){
//                gripper_tmp[j] = T(_gripper[i][j]);
//            }
//            ceres::QuaternionRotatePoint(q_pred,gripper_tmp,pred_3d[i]);
//            pred_3d[i][0] += t_pred[0]; pred_3d[i][1] += t_pred[1]; pred_3d[i][2] += t_pred[2];
//            dis[i] = pow(pred_3d[i][0]-_fullpoint[0],2)+pow(pred_3d[i][1]-_fullpoint[1],2)+pow(pred_3d[i][2]-_fullpoint[2],2);
//        }
//        auto time2 = system_clock::now();
//        sort(dis,dis+num);
//        auto time3 = system_clock::now();
//        auto dur1 = duration_cast<microseconds>(time2-time1);
//        auto dur2 = duration_cast<microseconds>(time3-time2);
//        cout<<'1'<<double(dur1.count()) * microseconds::period::num / microseconds::period::den
//        <<endl<<'2'<<double(dur2.count()) * microseconds::period::num / microseconds::period::den <<endl;
//        residual[0] = dis[0];
//
//        return true;
//    }
//
//    const double* _fullpoint;
//    const vector<vector<double>> _gripper;
//};

py::tuple optimize_step1_usedepth(
        const vector<vector<vector<double>>> &match2ds,     //(8,61?,2)
                           const vector<vector<vector<double>>> &match3ds,     //(8,61?,3)
                           const vector<vector<double>> &camera_ks,             //(8,4)
                            const vector<double> &q_pred,                      //(4)
                            const vector<double> &t_pred,                       //(3)
                            const vector<vector<vector<double>>> &keypoint      //8,61?,3
                           ) {
    double q[4];
    for (int i=0;i<4;++i){
        q[i] = q_pred[i];
    }

    double t[3];
    for (int i=0;i<3;++i){
        t[i] = t_pred[i];
    }

    unsigned long num_frames = match2ds.size();
    unsigned long num_points;

    // 构建最小二乘问题
    ceres::Problem problem;
    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::QuaternionParameterization();
    for (int i=0;i<num_frames;i++) {
        num_points = match2ds[i].size();
        for (int j=0;j<num_points;j++){
            problem.AddResidualBlock(     // 向问题中添加误差项
                    // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
                    new ceres::AutoDiffCostFunction<REPROJECTIONandKEY3D_COST, 5, 4, 3>(
                            new REPROJECTIONandKEY3D_COST(match2ds[i][j].data(),
                                                  match3ds[i][j].data(),
                                                  camera_ks[i].data(),
                                                  keypoint[i][j].data()
                                                  )
                    ),
                    nullptr,            // 核函数，这里不使用，为空
                    q,
                    t
            );
        }
    }
    problem.SetParameterization(q, quaternion_local_parameterization);

    // 配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimize time cost = " << time_used.count() << " seconds. " << endl;

    // 输出结果
//    cout << summary.BriefReport() << endl;

    return py::make_tuple(py::array_t<double>(4,q),
                   py::array_t<double>(3,t)
            );
}

py::tuple optimize_step1_nodepth(
        const vector<vector<vector<double>>> &match2ds,     //(8,61?,2)
        const vector<vector<vector<double>>> &match3ds,     //(8,61?,3)
        const vector<vector<double>> &camera_ks,             //(8,4)
        const vector<double> &q_pred,                      //(4)
        const vector<double> &t_pred                       //(3)
) {
    double q[4];
    for (int i=0;i<4;++i){
        q[i] = q_pred[i];
    }

    double t[3];
    for (int i=0;i<3;++i){
        t[i] = t_pred[i];
    }

    unsigned long num_frames = match2ds.size();
    unsigned long num_points;

    // 构建最小二乘问题
    ceres::Problem problem;
    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::QuaternionParameterization();
    for (int i=0;i<num_frames;i++) {
        num_points = match2ds[i].size();
        for (int j=0;j<num_points;j++){
            problem.AddResidualBlock(     // 向问题中添加误差项
                    // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
                    new ceres::AutoDiffCostFunction<REPROJECTION_COST, 2, 4, 3>(
                            new REPROJECTION_COST(match2ds[i][j].data(),
                                                          match3ds[i][j].data(),
                                                          camera_ks[i].data()
                            )
                    ),
                    nullptr,            // 核函数，这里不使用，为空
                    q,
                    t
            );
        }
    }
    problem.SetParameterization(q, quaternion_local_parameterization);

    // 配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimize time cost = " << time_used.count() << " seconds. " << endl;

    // 输出结果
//    cout << summary.BriefReport() << endl;

    return py::make_tuple(py::array_t<double>(4,q),
                          py::array_t<double>(3,t)
    );
}

// 这样写太慢
//py::tuple optimize_step2(
//        const vector<double> &q_pred,                      //(4)
//        const vector<double> &t_pred,                       //(3)
//        const vector<vector<vector<double>>> &full_depth_point,      //8,61?,3
//        const vector<vector<vector<double>>> &gripper_point     //8,8192,3
//) {
//    double q[4];
//    for (int i=0;i<4;++i){
//        q[i] = q_pred[i];
//    }
//
//    double t[3];
//    for (int i=0;i<3;++i){
//        t[i] = t_pred[i];
//    }
//
//    unsigned long num_frames = full_depth_point.size();
//    unsigned long num_points;
//
//    // 构建最小二乘问题
//    ceres::Problem problem;
//    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::QuaternionParameterization();
//    for (int i=0;i<num_frames;i++) {
//        num_points = full_depth_point[i].size();
//        for (int j=0;j<num_points;j++){
//            problem.AddResidualBlock(     // 向问题中添加误差项
//                    // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
//                    new ceres::AutoDiffCostFunction<FULL3D_COST, 1, 4, 3>(
//                            new FULL3D_COST(full_depth_point[i][j].data(),
//                                            gripper_point[i]
//                            )
//                    ),
//                    nullptr,            // 核函数，这里不使用，为空
//                    q,
//                    t
//            );
//        }
//    }
//    problem.SetParameterization(q, quaternion_local_parameterization);
//
//    // 配置求解器
//    ceres::Solver::Options options;     // 这里有很多配置项可以填
//    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
//    options.minimizer_progress_to_stdout = true;   // 输出到cout
//
//    ceres::Solver::Summary summary;                // 优化信息
//    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
//    ceres::Solve(options, &problem, &summary);  // 开始优化
//    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
//    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
//    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
//
//    // 输出结果
//    cout << summary.BriefReport() << endl;
//
//    return py::make_tuple(py::array_t<double>(4,q),
//                          py::array_t<double>(3,t)
//    );
//}

py::tuple adjust(
        const vector<vector<vector<double>>> &match2ds,     //(8,61?,2)
        const vector<vector<vector<double>>> &match3ds,     //(8,61?,3)
        const vector<vector<double>> &camera_ks,             //(8,4)
        const vector<vector<double>> &q_preds,                      //(8,4)
        const vector<vector<double>> &t_preds                       //(8,3)
) {
    unsigned long num_frames = match2ds.size();
    vector<double*> qs(num_frames);
    vector<double*> ts(num_frames);

    vector<double> q_data(num_frames * 4);
    vector<double> t_data(num_frames * 3);
    for (int i = 0; i < num_frames; ++i) {
        for (int j = 0; j < 4; ++j) {
            q_data[i * 4 + j] = q_preds[i][j];
        }
        for (int j = 0; j < 3; ++j) {
            t_data[i * 3 + j] = t_preds[i][j];
        }
    }


    unsigned long num_points;

    // 构建最小二乘问题
    ceres::Problem problem;
    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::QuaternionParameterization();
    // 向问题中添加参数化变量
    for (int i = 0; i < num_frames; ++i) {
        problem.AddParameterBlock(&q_data[i * 4], 4, quaternion_local_parameterization);
        problem.AddParameterBlock(&t_data[i * 3], 3);
    }
    for (int i=0;i<num_frames;i++) {
        num_points = match2ds[i].size();
        for (int j=0;j<num_points;j++){
            problem.AddResidualBlock(     // 向问题中添加误差项
                    // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
                    new ceres::AutoDiffCostFunction<REPROJECTION_COST, 2, 4, 3>(
                            new REPROJECTION_COST(match2ds[i][j].data(),
                                                  match3ds[i][j].data(),
                                                  camera_ks[i].data()
                            )
                    ),
                    nullptr,            // 核函数，这里不使用，为空
                    &q_data[i * 4],
                    &t_data[i * 3]
            );
        }
    }


    // 配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   //

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve(options, &problem, &summary);  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "adjust time cost = " << time_used.count() << " seconds. " << endl;

    // 输出结果
//    cout << summary.BriefReport() << endl;

    return py::make_tuple(q_data,t_data);
}


PYBIND11_MODULE(try_pybind11,m){
    m.def("optimize_step1_usedepth",&optimize_step1_usedepth);
    m.def("optimize_step1_nodepth",&optimize_step1_nodepth);
    m.def("adjust",&adjust);
}