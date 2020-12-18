//
// Created by v_b1xian on 2020-11-26.
//
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#ifndef VISION_PEOFILE_TEST_H
#define VISION_PEOFILE_TEST_H

using namespace std;

namespace vacv {


class CvProfile {

public:
    using TestFunc = std::function<std::vector<double>()>;
    using TestFuncInfo = std::pair<TestFunc, std::string>;
    using TestFuncList = std::vector<TestFuncInfo>;
    using SpeedResult = std::pair<std::vector<double>, std::string>;
    using OutputResult = std::pair<std::vector<double>, std::string>;
    static void profile(TestFuncList& func_list,
            const TestFunc& setup_func,
            const TestFunc& clean_func,
            std::vector<SpeedResult>& speed_profile,
            std::vector<OutputResult>& output_profile
            );

private:
    static void print_results(std::vector<SpeedResult>& speed_result,
                              std::vector<OutputResult>& output_result);
    static void save_results(std::string tag, std::vector<long>& records);
    static int _k_test_times;

};

} // namespace vacv

#endif //VISION_PEOFILE_TEST_H
