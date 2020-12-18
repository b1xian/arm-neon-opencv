#ifndef PERF_UTIL_H
#define PERF_UTIL_H

#include <map>
#include <string>

#define TIME_PERF(duration) AutoPerf perf(duration)

class AutoPerf {
public:
    AutoPerf(double& duration);
    ~AutoPerf();

private:
    double* _duration;
    clock_t _start;
};

#endif //PERF_UTIL_H
