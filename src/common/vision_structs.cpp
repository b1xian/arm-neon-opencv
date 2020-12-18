//
// Created by Li,Wendong on 2019-01-03.
//
#include "vision_structs.h"
#include <cstring>

namespace vision {

    // For VPoint2D ------------------------------------------
    void VPoint::copy(const VPoint &info) {
        x = info.x;
        y = info.y;
    }

    void VPoint::clear() {
        x = 0.0F;
        y = 0.0F;
    }

    VPoint operator+(const VPoint &p1, const VPoint &p2) {
        VPoint p(p1);
        p += p2;
        return p;
    }

    VPoint operator-(const VPoint &p1, const VPoint &p2) {
        VPoint p(p1);
        p -= p2;
        return p;
    }

    void VPoint::operator+=(const VPoint &p) {
        x += p.x;
        y += p.y;
    }

    void VPoint::operator-=(const VPoint &p) {
        x -= p.x;
        y -= p.y;
    }

    void VPoint::operator/=(float val) {
        x /= val;
        y /= val;
    }

    // For VPoint3D ------------------------------------------
    void VPoint3::copy(const VPoint3 &info) {
        x = info.x;
        y = info.y;
        z = info.z;
    }

    void VPoint3::clear() {
        x = 0.0F;
        y = 0.0F;
        z = 0.0F;
    }

    void VAngle::copy(const VAngle &info) {
        yaw = info.yaw;
        pitch = info.pitch;
        roll = info.roll;
    }

    void VAngle::clear() {
        yaw = 0.0;
        pitch = 0.0;
        roll = 0.0;
    }

    void VRect::set(float l, float t, float r, float b) {
        left = l;
        top = t;
        right = r;
        bottom = b;
    }

    float VRect::width() const {
        return right - left;
    }

    float VRect::height() const {
        return bottom - top;
    }

    bool VRect::contains(float x, float y) {
        return left < right && top < bottom  // check for empty first
               && x >= left && x < right && y >= top && y < bottom;
    }

    void VEyeInfo::copy(const VEyeInfo &info) {
        x = info.x;
        y = info.y;
        width = info.width;
        height = info.height;
    }

    void VEyeInfo::clear() {
        x = 0;
        y = 0;
        width = 0;
        height = 0;
    }

    void VMatrix::copy(const VMatrix &info) {
        x = info.x;
        y = info.y;
        z = info.z;
    }

    void VMatrix::clear() {
        x = 0.0;
        y = 0.0;
        z = 0.0;
    }

    void VState::clear() {
        state = -1;
        continue_time = -1;
        trigger_count = -1;
    }

    void VState::copy(const VState &info) {
        state = info.state;
        continue_time = info.continue_time;
        trigger_count = info.trigger_count;
    }

} // namespace vision