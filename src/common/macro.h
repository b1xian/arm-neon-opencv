#ifndef VISION_MACRO_H
#define VISION_MACRO_H

#include <climits>

/// math
#ifndef MIN
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#endif

#ifndef MAX
#define MAX(x, y) ((x) > (y) ? (x) : (y))
#endif

#ifndef CLAMP
#define CLAMP(x, min, max) MIN(MAX(x, min), max)
#endif

/// CONVERT
#define VA_TO_INT(value) static_cast<int>(value)
#define VA_TO_SHORT(value) static_cast<short>(value)
#define VA_TO_FLOAT(value) static_cast<float>(value)
#define VA_F_TO_BOOL(value) (value > 1e-6)
#define VA_I_TO_BOOL(value) (value == 1)
#ifndef SATURATE_CAST_SHORT
#define SATURATE_CAST_SHORT(X)                                               \
short MIN(                                                       \
  MAX(static_cast<int>(X + (X >= 0.f ? 0.5f : -0.5f)), SHRT_MIN), \
  SHRT_MAX);
#endif

#endif //VISION_MACRO_H
