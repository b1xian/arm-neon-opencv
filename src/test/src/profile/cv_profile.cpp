#include "cv_profile.h"

#include <algorithm>
#include <assert.h>
#include <iomanip>
#include <iostream>
#include <math.h>


#define MAX_DIFF 5e-4

namespace vacv {

int CvProfile::_k_test_times = 10;
static int k_log_batch_size =  5;
static const char* TAG = "CvProfiler";

void CvProfile::profile(TestFuncList& func_list,
                        const TestFunc& setup_func,
                        const TestFunc& clean_func,
                        std::vector<SpeedResult>& speed_profile,
                        std::vector<OutputResult>& output_profile) {
    if (func_list.empty()) {
        return;
    }

    std::cout << std::setw(30) << std::setfill('=') << " " << std::endl;
    std::cout << "CV Profiler Starts..." << std::endl;
    std::cout << "Test function number: " << func_list.size() << std::endl;
    std::cout << "Test times: " << _k_test_times << std::endl;
    std::cout << std::setw(30) << std::setfill('=') << " " << std::endl;

    auto func_num = func_list.size();
    speed_profile.resize(func_num);
    output_profile.resize(func_num);

    std::vector<double> output_consine_distance(func_num);
    std::vector<double> expect_consine_distance(func_num);
    std::vector<double> total_durations_opencv(func_num);
    std::vector<double> total_durations_vacv(func_num);

    for (int i = 0; i < _k_test_times; ++i) {
        if (setup_func) {
            setup_func();
        }

        int index = 0;
        for (const auto& func : func_list) {
            std::vector<double> profile_details = func.first();

            total_durations_opencv[index] += profile_details[0];
            total_durations_vacv[index] += profile_details[1];
            output_consine_distance[index] += profile_details[2];
            expect_consine_distance[index] = profile_details[3];
            index++;
        }

        if ((i + 1) % k_log_batch_size == 0) {
            std::cout << std::endl;
            for (int idx = 0; idx < func_num; ++idx) {
                std::cout << "[" << TAG << "] func=" << func_list[idx].second
                          << ", batch=" << (i + 1) / k_log_batch_size
                          << ", avg_opencv_duration=" << total_durations_opencv[idx] / (i + 1) << " ms"
                          << ", avg_vacv_duration=" << total_durations_vacv[idx] / (i + 1) << " ms"
                          << std::endl;
            }
        }

        if (clean_func) {
            clean_func();
        }
    }

    for (int i = 0; i < func_num; i++) {
        double func_durations_opencv = total_durations_opencv[i];
        double func_durations_vacv = total_durations_vacv[i];
        double func_output_consine_distance = output_consine_distance[i];
        double func_expect_consine_distance = expect_consine_distance[i];

        auto func_tag = func_list[i].second;

        std::vector<double> speed_profile_details(2);
        speed_profile_details[0] = func_durations_opencv / _k_test_times;
        speed_profile_details[1] = func_durations_vacv / _k_test_times;
        speed_profile[i] = {speed_profile_details, func_tag};


        std::vector<double> output_details(2);
        output_details[0] = func_output_consine_distance / _k_test_times;
        output_details[1] = func_expect_consine_distance;
        output_profile[i] = {output_details, func_tag};
    }

    print_results(speed_profile, output_profile);
}

void CvProfile::print_results(std::vector<SpeedResult>& speed_profile, std::vector<OutputResult>& output_result) {
    std::cout << std::setw(80) << std::setfill('=') << " " << std::endl;
    std::cout.precision(4);
    for (int i = 0; i < speed_profile.size(); i++) {
        SpeedResult profile = speed_profile[i];
        OutputResult output = output_result[i];
        std::cout << std::left << std::setw(40) << std::setfill(' ') << profile.second
                  << "opencv: " << std::setw(3) << std::setw(3) << profile.first[0] << " ms\t"
                  << "vacv: " << std::setw(3) << std::setw(3) << profile.first[1] << " ms\t"
                  << "(output:" << output.first[0] << ", expect:" << output.first[1] - MAX_DIFF << ")"
                  << std::endl;
        double diff = abs(output.first[0] - output.first[1]);
        if (diff > MAX_DIFF) {
            printf("\033[01;31m 【TEST FAILED!】\033[0m\n");
        } else {
            printf("\033[01;32m 【TEST SUCCESS】\033[0m\n");
        }

    }
    std::cout << std::right << std::setw(80) << std::setfill('=') << " " << std::endl << std::endl;
}

void CvProfile::save_results(std::string tag, std::vector<long>& records) {
    // todo
}

} // namespace xperf