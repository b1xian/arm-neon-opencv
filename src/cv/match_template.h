#ifndef VISION_MATCH_TEMPLATE_H
#define VISION_MATCH_TEMPLATE_H

#include "../common/tensor.h"

namespace va_cv {

class MatchTemplate {
public:
    static void match_template(const vision::Tensor& src, const vision::Tensor& target,
                               vision::Tensor& result, int method);

    static void minMaxIdx(const vision::Tensor& src, double* minVal, double* maxVal,
                   int* minIdx = nullptr, int* maxIdx = nullptr, const vision::Tensor& mask = vision::Tensor());

private:
    static void match_template_opencv(const vision::Tensor& src, const vision::Tensor& target,
                                      vision::Tensor& result, int method);

    static void match_template_naive(const vision::Tensor& src, const vision::Tensor& target,
                                     vision::Tensor& result, int method);

    static void match_template_sse(const vision::Tensor& src, const vision::Tensor& target,
                                   vision::Tensor& result, int method);

    static void match_template_neon(const vision::Tensor& src, const vision::Tensor& target,
                                    vision::Tensor& result, int method);
};

} // namespace va_cv

#endif //VISION_MATCH_TEMPLATE_H
