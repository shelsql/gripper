#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <ceres/jet.h>
using namespace std;
using namespace chrono;
namespace py = pybind11;

void UnitQuaternionRotatePoint(const double q[4], const double pt[3], double result[3]) {
    const double t2 =  q[0] * q[1];
    const double t3 =  q[0] * q[2];
    const double t4 =  q[0] * q[3];
    const double t5 = -q[1] * q[1];
    const double t6 =  q[1] * q[2];
    const double t7 =  q[1] * q[3];
    const double t8 = -q[2] * q[2];
    const double t9 =  q[2] * q[3];
    const double t1 = -q[3] * q[3];
    result[0] = double(2) * ((t8 + t1) * pt[0] + (t6 - t4) * pt[1] + (t3 + t7) * pt[2]) + pt[0];  // NOLINT
    result[1] = double(2) * ((t4 + t6) * pt[0] + (t5 + t1) * pt[1] + (t9 - t2) * pt[2]) + pt[1];  // NOLINT
    result[2] = double(2) * ((t7 - t3) * pt[0] + (t2 + t9) * pt[1] + (t5 + t8) * pt[2]) + pt[2];  // NOLINT
}
void QuaternionRotatePoint(const double q[4], const double pt[3], double result[3]) {
    // 'scale' is 1 / norm(q).
    const double scale = double(1) / sqrt(q[0] * q[0] +
                                q[1] * q[1] +
                                q[2] * q[2] +
                                q[3] * q[3]);

    // Make unit-norm version of q.
    const double unit[4] = {
            scale * q[0],
            scale * q[1],
            scale * q[2],
            scale * q[3],
    };

    UnitQuaternionRotatePoint(unit, pt, result);
}

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
    ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
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
                    loss_function ,            // 核函数，这里不使用，为空
                    q,
                    t
            );
        }
    }
    problem.AddParameterBlock(q,4);
    problem.SetParameterization(q, quaternion_local_parameterization);

    // 配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = false;   // 输出到cout

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
    ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
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
                    loss_function,            // 核函数，这里不使用，为空
                    q,
                    t
            );
        }
    }
    problem.SetParameterization(q, quaternion_local_parameterization);

    // 配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = false;   // 输出到cout

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

struct ADJUST_COST {
    ADJUST_COST(              const double* match3d,  // 3  points in frame i
                              const double* camera_params,     // 4 fx,cx,fy,cy 无所谓，反正都一样
                              const double* depth,    // 360*640     depth image of frame j
                              const double* keypoint    // 3
    ) :
             _match3d(match3d), _camera_params(camera_params),_depth(depth),_keypoint(keypoint)
    {}

    // 残差的计算
    template<typename T>                // T相当于一个数据类型，只不过不能把它写的具体
    bool operator()(                    // 重载括号运算符 类就相当于一个函数了   在这里实际是给人家内部自动求导用的
            const T *const qi,          // 待优化四元数及位移
            const T *const ti,
            const T *const qj,
            const T *const tj,
            T *residual) const {
        T key3d[3];
        T key2d[2];

        T qi_inv[4];
        qi_inv[0] = qi[0];
        for (int i=1;i<4;++i){
            qi_inv[i] = -qi[i];
        }
        T qj_inv[4];
        qj_inv[0] = qj[0];
        for (int i=1;i<4;++i){
            qj_inv[i] = -qj[i];
        }

        //搞不懂为啥非得这么写
        T keypoint_tmp[3];  // p
        for (int i=0;i<3;++i) {
            keypoint_tmp[i] = T(_keypoint[i]);
        }
        // 左乘Ti逆
        ceres::QuaternionRotatePoint(qi_inv,keypoint_tmp,key3d);
        T t_tmp1[3];
        ceres::QuaternionRotatePoint(qi_inv,ti,t_tmp1);
        key3d[0] -= t_tmp1[0]; key3d[1] -= t_tmp1[1]; key3d[2] -= t_tmp1[2];
        // 调试
//        ceres::QuaternionRotatePoint(qi,key3d,key3d);
//        key3d[0] += ti[0];key3d[1] += ti[1];key3d[2] += ti[2];
//        cout<<"diff"<<ceres::Jet<double, 14>(key3d[0]-keypoint_tmp[0])<<endl<<ceres::Jet<double, 14>(key3d[0]-keypoint_tmp[0])<<endl;
        // 调试



        // 左乘Tj
        ceres::QuaternionRotatePoint(qj,key3d,key3d);
        key3d[0]+=tj[0];key3d[1]+=tj[1];key3d[2]+=tj[2];

        // 投影
        key2d[0] = (key3d[0]/key3d[2])*T(_camera_params[0]) + T(_camera_params[1]);
        key2d[1] = (key3d[1]/key3d[2])*T(_camera_params[2]) + T(_camera_params[3]);

        // 用frame j的深度图恢复到3d
        T key3d1[3];

        int idx = ceres::Jet<double, 14>(key2d[0]).a;
        int idy = ceres::Jet<double, 14>(key2d[1]).a;
        if (idx > 639){
            idx = 639;
        }
        if(idx<0){
            idx = 0;
        }
        if(idy>359){
            idy=359;
        }
        if(idy<0){
            idy=0;
        }
        cout<<idx<<"and"<<idy<<endl;
        key3d1[2] = T(_depth[idy*640+idx]);
        key3d1[0] = key3d1[2]*(key2d[0]- T(_camera_params[1]))/ T(_camera_params[0]);
        key3d1[1] = key3d1[2]*(key2d[1]- T(_camera_params[3]))/ T(_camera_params[2]);
        cout<<"key3d1"<<ceres::Jet<double, 14>(key3d1[2]).a<<"and"<<ceres::Jet<double, 14>(key3d1[0]).a<<"and"<<ceres::Jet<double, 14>(key3d1[1]).a<<endl;


        T key3d2[3];
        // 左乘Tj逆
        ceres::QuaternionRotatePoint(qj_inv,key3d1,key3d2);
        T t_tmp2[3];
        ceres::QuaternionRotatePoint(qj_inv,tj,t_tmp2);
        key3d2[0] -= t_tmp2[0]; key3d2[1] -= t_tmp2[1]; key3d2[2] -= t_tmp2[2];

        T key3d3[3];
        // 左乘Ti
        ceres::QuaternionRotatePoint(qi,key3d1,key3d3);
        key3d3[0]+=ti[0];key3d3[1]+=ti[1];key3d3[2]+=ti[2];
        cout<<"key3d3"<<ceres::Jet<double, 14>(key3d3[2]).a<<"and"<<ceres::Jet<double, 14>(key3d3[0]).a<<"and"<<ceres::Jet<double, 14>(key3d3[1]).a<<endl;

        if(ceres::Jet<double, 14>(_depth[idy*640+idx]).a <= 0.0){
            // 用深度图计算keypoint的距离
            residual[0] = 0.0*(key3d3[0] - T(keypoint_tmp[0]));
            residual[1] = 0.0*(key3d3[1] - T(keypoint_tmp[1]));
            residual[2] = 0.0*(key3d3[2] - T(keypoint_tmp[2]));
        }
        else{
            // 用深度图计算keypoint的距离
            residual[0] = 1.0*(key3d3[0] - T(keypoint_tmp[0]));
            residual[1] = 1.0*(key3d3[1] - T(keypoint_tmp[1]));
            residual[2] = 1.0*(key3d3[2] - T(keypoint_tmp[2]));
        }



        return true;
    }

    const double* _match3d;
    const double* _camera_params;
    const double* _depth;
    const double* _keypoint;
};

py::tuple adjust(
//        const vector<vector<vector<double>>> &match2ds,     //(8,61?,2)
        const vector<vector<vector<double>>> &match3ds,     //(8,61?,3)
        const vector<vector<double>> &camera_ks,             //(8,4)
        const vector<vector<double>> &q_preds,                      //(8,4)
        const vector<vector<double>> &t_preds,                       //(8,3)
        const vector<vector<vector<double>>> &keypoint,      //8,61?,3
        const vector<vector<double>> &depths                //(8,360*640)
) {
    unsigned long num_frames = match3ds.size();
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

    double q_tmp[4];
    for(int i=0;i<4;++i){
        q_tmp[i] = q_preds[0][i];
    }
    double t_tmp[3];
    for(int i=0;i<3;++i){
        t_tmp[i] = t_preds[0][i];
    }

    unsigned long num_points;

    // 构建最小二乘问题
    ceres::LossFunction* loss_function = new ceres::CauchyLoss(1.0);
    ceres::Problem problem;
    ceres::LocalParameterization* quaternion_local_parameterization = new ceres::QuaternionParameterization();
    // 向问题中添加参数化变量
    for (int i = 0; i < num_frames; ++i) {
        problem.AddParameterBlock(&q_data[i * 4], 4, quaternion_local_parameterization);
        problem.AddParameterBlock(&t_data[i * 3], 3);
    }
    for (int i=0;i<num_frames;i++) {
        num_points = match3ds[i].size();
        for (int j=0;j<num_frames;j++){
//            if(i==j){
//                continue;
//            }
            for(int k=0;k<num_points;++k){
                problem.AddResidualBlock(     // 向问题中添加误差项
                        // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
                        new ceres::AutoDiffCostFunction<ADJUST_COST, 3, 4, 3,4,3>(
                                new ADJUST_COST(match3ds[i][k].data(),
                                                camera_ks[i].data(),
                                                depths[j].data(),
                                                keypoint[i][k].data()
                                )
                        ),
                        nullptr,//loss_function,            // 核函数，这里不使用，为空
                        &q_data[i*4],
                        &t_data[i*3],
//                        &q_data[j*4],
//                        &t_data[j*3]
                            q_tmp,
                            t_tmp
                );
            }
        }
    }


    // 配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = false;   //

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
//    m.def("adjust",&adjust);
}