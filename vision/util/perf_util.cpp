#include "perf_util.h"


AutoPerf::AutoPerf(double& duration)
        : _duration(&duration){
    _start = clock();
}

AutoPerf::~AutoPerf() {
    if (_duration) {
        *_duration = (double)(clock() - _start) / CLOCKS_PER_SEC * 1000;
    }
}
