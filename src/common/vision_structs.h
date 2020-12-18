#ifndef VISION_STRUCT_H
#define VISION_STRUCT_H

namespace vision {

    class VPoint {
    public:
        VPoint(float _x, float _y) {
            x = _x;
            y = _y;
        }

        VPoint() {
            x = 0.0F;
            y = 0.0F;
        }

        void copy(const VPoint &info);

        void clear();

        friend VPoint operator+(const VPoint &p1, const VPoint &p2);

        friend VPoint operator-(const VPoint &p1, const VPoint &p2);

        void operator+=(const VPoint &p);

        void operator-=(const VPoint &p);

        void operator/=(float val);

        float x;
        float y;
    };

    class VPoint3 {
    public:
        VPoint3(float _x, float _y, float _z) {
            x = _x;
            y = _y;
            z = _z;
        }

        VPoint3() {
            x = 0.0;
            y = 0.0;
            z = 0.0;
        }

        void copy(const VPoint3 &info);

        void clear();

        float x;
        float y;
        float z;
    };

    class VAngle {
    public:
        VAngle(float x, float y, float z) {
            yaw = x;
            pitch = y;
            roll = z;
        }

        VAngle() {
            yaw = 0.0F;
            pitch = 0.0F;
            roll = 0.0F;
        }

        void copy(const VAngle &info);

        void clear();

        float yaw;
        float pitch;
        float roll;
    };

    class VEyeInfo {
    public:
        VEyeInfo(float _x, float _y, float _width, float _height) : x(_x), y(_y), width(_width), height(_height) {
        }

        VEyeInfo() : x(0), y(0), width(0), height(0) {
        }

        void copy(const VEyeInfo &info);

        void clear();

        float x, y, width, height;
        VPoint _eye_center;
        VPoint _eye_centroid;
    };

    class VMatrix {
    public:
        VMatrix(float _x, float _y, float _z) {
            x = _x;
            y = _y;
            z = _z;
        }

        VMatrix() {
            x = 0.0F;
            y = 0.0F;
            z = 0.0F;
        }

        void copy(const VMatrix &info);

        void clear();

        float x;
        float y;
        float z;
    };

    struct VRect {
        float left;
        float top;
        float right;
        float bottom;
        VRect(float _left, float _top, float _right, float _bottom)
           : left(_left), top(_top), right(_right), bottom(_bottom) {}
        void set(float left, float top, float right, float bottom);
        float width() const;
        float height() const;
        bool contains(float x, float y);
    };

    // 简单的size
    struct SimpleSize {
    public:
        float width;
        float height;
    };
    // 坐标极值
    struct ExtreSize {
        int x_min;
        int y_min;
        int x_max;
        int y_max;
    };
    // 记录索引和值
    struct IndexValue {
        int index;
        float value;
    };

    enum model_class_type {
        param = 1,
        bin = 2,
        txt = 3
    };

    enum VSlidingState {
        NON,     // 没有检测到
        START,   // 开始
        ONGOING, // 持续
        END      // 结束
    };

    struct VState {
        int state; // 0-没有监测到, 1-开始，2-持续，3-中断
        int continue_time; // 持续的时间
        int trigger_count; // 触发的次数

        VState() { clear(); }

        void clear();

        void copy(const VState &info);
    };

    struct VisGesture {
        int label;
        float confidence;
        float x1;
        float y1;
        float x2;
        float y2;
    };

    // 均值算法
    enum NORMAL_ALG {
        MUL, DIV
    };

} // namespace vision

#endif // VISION_STRUCT_H











