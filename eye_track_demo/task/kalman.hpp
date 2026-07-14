#pragma once

#include <algorithm>
#include <array>
#include <cmath>

#include "types.hpp"

namespace eye_track {

class ConstantVelocityKalman2D {
public:
    bool initialized() const { return initialized_; }

    void Reset() {
        initialized_ = false;
        state_.fill(0.f);
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) covariance_[r][c] = 0.f;
    }

    void Initialize(float x, float y, TimePoint now,
                    float position_variance,
                    float velocity_variance) {
        state_[0] = x;
        state_[1] = y;
        state_[2] = 0.f;
        state_[3] = 0.f;
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c) covariance_[r][c] = 0.f;
        covariance_[0][0] = std::max(1e-6f, position_variance);
        covariance_[1][1] = std::max(1e-6f, position_variance);
        covariance_[2][2] = std::max(1e-6f, velocity_variance);
        covariance_[3][3] = std::max(1e-6f, velocity_variance);
        last_time_ = now;
        initialized_ = true;
    }

    void Predict(TimePoint now, float process_noise, float max_dt = 0.1f) {
        if (!initialized_) return;
        float dt = std::chrono::duration<float>(now - last_time_).count();
        dt = std::max(1.f / 500.f, std::min(max_dt, dt));
        state_[0] += state_[2] * dt;
        state_[1] += state_[3] * dt;

        float transition[4][4] = {
            {1.f, 0.f, dt, 0.f},
            {0.f, 1.f, 0.f, dt},
            {0.f, 0.f, 1.f, 0.f},
            {0.f, 0.f, 0.f, 1.f}
        };
        float temp[4][4] = {{0.f}};
        float next[4][4] = {{0.f}};
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                for (int k = 0; k < 4; ++k)
                    temp[r][c] += transition[r][k] * covariance_[k][c];
            }
        }
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                for (int k = 0; k < 4; ++k)
                    next[r][c] += temp[r][k] * transition[c][k];
            }
        }

        const float q = std::max(0.f, process_noise);
        const float dt2 = dt * dt;
        const float dt3 = dt2 * dt;
        const float dt4 = dt2 * dt2;
        const float q_pos = 0.25f * dt4 * q;
        const float q_cross = 0.5f * dt3 * q;
        const float q_vel = dt2 * q;
        next[0][0] += q_pos;
        next[0][2] += q_cross;
        next[2][0] += q_cross;
        next[2][2] += q_vel;
        next[1][1] += q_pos;
        next[1][3] += q_cross;
        next[3][1] += q_cross;
        next[3][3] += q_vel;

        CopyCovariance(next);
        last_time_ = now;
    }

    void Correct(float x, float y, float measurement_variance) {
        if (!initialized_) return;
        const float r = std::max(1e-6f, measurement_variance);
        const float s00 = covariance_[0][0] + r;
        const float s01 = covariance_[0][1];
        const float s10 = covariance_[1][0];
        const float s11 = covariance_[1][1] + r;
        const float det = s00 * s11 - s01 * s10;
        if (std::fabs(det) < 1e-12f) return;

        const float inv00 = s11 / det;
        const float inv01 = -s01 / det;
        const float inv10 = -s10 / det;
        const float inv11 = s00 / det;

        float gain[4][2] = {{0.f}};
        for (int i = 0; i < 4; ++i) {
            gain[i][0] = covariance_[i][0] * inv00 +
                         covariance_[i][1] * inv10;
            gain[i][1] = covariance_[i][0] * inv01 +
                         covariance_[i][1] * inv11;
        }

        const float residual_x = x - state_[0];
        const float residual_y = y - state_[1];
        for (int i = 0; i < 4; ++i)
            state_[i] += gain[i][0] * residual_x + gain[i][1] * residual_y;

        float next[4][4] = {{0.f}};
        for (int r_index = 0; r_index < 4; ++r_index) {
            for (int c = 0; c < 4; ++c) {
                next[r_index][c] = covariance_[r_index][c] -
                    gain[r_index][0] * covariance_[0][c] -
                    gain[r_index][1] * covariance_[1][c];
            }
        }
        CopyCovariance(next);
    }

    void ClampPosition(float min_x, float min_y, float max_x, float max_y) {
        if (!initialized_) return;
        const float clamped_x = std::max(min_x, std::min(max_x, state_[0]));
        const float clamped_y = std::max(min_y, std::min(max_y, state_[1]));
        if (clamped_x != state_[0]) state_[2] = 0.f;
        if (clamped_y != state_[1]) state_[3] = 0.f;
        state_[0] = clamped_x;
        state_[1] = clamped_y;
    }

    Point2f Position() const { return Point2f{{state_[0], state_[1]}}; }
    Point2f Velocity() const { return Point2f{{state_[2], state_[3]}}; }

private:
    void CopyCovariance(float source[4][4]) {
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                covariance_[r][c] = 0.5f * (source[r][c] + source[c][r]);
            }
        }
        for (int i = 0; i < 4; ++i)
            covariance_[i][i] = std::max(covariance_[i][i], 1e-9f);
    }

    bool initialized_ = false;
    std::array<float, 4> state_{{0.f, 0.f, 0.f, 0.f}};
    float covariance_[4][4] = {{0.f}};
    TimePoint last_time_;
};

}  // namespace eye_track
