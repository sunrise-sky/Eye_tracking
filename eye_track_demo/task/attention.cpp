#include "attention.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

namespace eye_track {
namespace {

float Clamp01(float value) {
    return std::max(0.f, std::min(1.f, value));
}

const Point2f kCalibrationPoints[9] = {
    {{0.1f, 0.1f}}, {{0.5f, 0.1f}}, {{0.9f, 0.1f}},
    {{0.1f, 0.5f}}, {{0.5f, 0.5f}}, {{0.9f, 0.5f}},
    {{0.1f, 0.9f}}, {{0.5f, 0.9f}}, {{0.9f, 0.9f}}
};

class GazeCalibrator {
public:
    static const int kSamplesPerPoint = 30;

    void Start() {
        calibrating_ = true;
        calibrated_ = false;
        point_index_ = 0;
        sample_count_ = 0;
        sample_x_ = 0.f;
        sample_y_ = 0.f;
        printf("[Calib] Start calibration\n");
    }

    void Reset() {
        calibrating_ = false;
        calibrated_ = false;
        point_index_ = 0;
        sample_count_ = 0;
        sample_x_ = 0.f;
        sample_y_ = 0.f;
        scale_x_ = 8.f;
        scale_y_ = 8.f;
        offset_x_ = 0.f;
        offset_y_ = 0.f;
    }

    void AddSample(float x, float y) {
        if (!calibrating_) return;
        sample_x_ += x;
        sample_y_ += y;
        sample_count_++;
        if (sample_count_ < kSamplesPerPoint) return;

        samples_[point_index_] = Point2f{{
            sample_x_ / kSamplesPerPoint,
            sample_y_ / kSamplesPerPoint}};
        printf("[Calib] Point %d done\n", point_index_ + 1);
        point_index_++;
        sample_count_ = 0;
        sample_x_ = 0.f;
        sample_y_ = 0.f;
        if (point_index_ >= 9) {
            Fit();
            calibrating_ = false;
            calibrated_ = true;
            printf("[Calib] Done! scale=(%.2f,%.2f)\n", scale_x_, scale_y_);
        }
    }

    Point2f Map(float x, float y) const {
        return Point2f{{
            Clamp01(0.5f + (x - offset_x_) * scale_x_),
            Clamp01(0.5f + (y - offset_y_) * scale_y_)}};
    }

    CalibrationState State() const {
        CalibrationState state;
        state.active = calibrating_;
        state.calibrated = calibrated_;
        state.point_index = std::min(point_index_, 8);
        state.sample_count = sample_count_;
        if (calibrating_ && point_index_ < 9)
            state.target = kCalibrationPoints[point_index_];
        return state;
    }

    bool calibrating() const { return calibrating_; }

private:
    void Fit() {
        float sum_x = 0.f;
        float sum_y = 0.f;
        float sum_target_x = 0.f;
        float sum_target_y = 0.f;
        float sum_x2 = 0.f;
        float sum_y2 = 0.f;
        float sum_x_target = 0.f;
        float sum_y_target = 0.f;
        for (int i = 0; i < 9; ++i) {
            const float x = samples_[i][0];
            const float y = samples_[i][1];
            const float target_x = kCalibrationPoints[i][0] - 0.5f;
            const float target_y = kCalibrationPoints[i][1] - 0.5f;
            sum_x += x;
            sum_y += y;
            sum_target_x += target_x;
            sum_target_y += target_y;
            sum_x2 += x * x;
            sum_y2 += y * y;
            sum_x_target += x * target_x;
            sum_y_target += y * target_y;
        }
        const float n = 9.f;
        const float denominator_x = n * sum_x2 - sum_x * sum_x;
        const float denominator_y = n * sum_y2 - sum_y * sum_y;
        if (std::fabs(denominator_x) > 1e-6f)
            scale_x_ = (n * sum_x_target - sum_x * sum_target_x) /
                       denominator_x;
        if (std::fabs(denominator_y) > 1e-6f)
            scale_y_ = (n * sum_y_target - sum_y * sum_target_y) /
                       denominator_y;
        offset_x_ = (sum_x - scale_x_ * sum_target_x) / n;
        offset_y_ = (sum_y - scale_y_ * sum_target_y) / n;
    }

    bool calibrating_ = false;
    bool calibrated_ = false;
    int point_index_ = 0;
    int sample_count_ = 0;
    float sample_x_ = 0.f;
    float sample_y_ = 0.f;
    float scale_x_ = 8.f;
    float scale_y_ = 8.f;
    float offset_x_ = 0.f;
    float offset_y_ = 0.f;
    Point2f samples_[9];
};

class LowPassFilter {
public:
    float Filter(float value, float alpha) {
        alpha = Clamp01(alpha);
        if (!initialized_) {
            value_ = value;
            initialized_ = true;
        } else {
            value_ = alpha * value + (1.f - alpha) * value_;
        }
        return value_;
    }

    void Reset() {
        initialized_ = false;
        value_ = 0.f;
    }

private:
    bool initialized_ = false;
    float value_ = 0.f;
};

class OneEuroFilter {
public:
    OneEuroFilter(float min_cutoff, float beta, float derivative_cutoff)
        : min_cutoff_(min_cutoff), beta_(beta),
          derivative_cutoff_(derivative_cutoff) {}

    float Filter(float value, TimePoint now) {
        if (!initialized_) {
            initialized_ = true;
            last_time_ = now;
            last_raw_ = value;
            derivative_filter_.Reset();
            value_filter_.Reset();
            derivative_filter_.Filter(0.f, 1.f);
            return value_filter_.Filter(value, 1.f);
        }
        float dt = std::chrono::duration<float>(now - last_time_).count();
        dt = std::max(1.f / 500.f, std::min(0.1f, dt));
        const float derivative = (value - last_raw_) / dt;
        const float filtered_derivative = derivative_filter_.Filter(
            derivative, Alpha(dt, derivative_cutoff_));
        const float cutoff = min_cutoff_ + beta_ * std::fabs(filtered_derivative);
        const float filtered = value_filter_.Filter(value, Alpha(dt, cutoff));
        last_time_ = now;
        last_raw_ = value;
        return filtered;
    }

    void Reset() {
        initialized_ = false;
        derivative_filter_.Reset();
        value_filter_.Reset();
    }

private:
    static float Alpha(float dt, float cutoff) {
        const float tau = 1.f / (2.f * 3.14159265358979323846f *
                                 std::max(0.01f, cutoff));
        return 1.f / (1.f + tau / dt);
    }

    float min_cutoff_;
    float beta_;
    float derivative_cutoff_;
    bool initialized_ = false;
    float last_raw_ = 0.f;
    TimePoint last_time_;
    LowPassFilter derivative_filter_;
    LowPassFilter value_filter_;
};

GazeDirection ClassifyGaze(float x, float y) {
    const bool left = x < 0.35f;
    const bool right = x > 0.65f;
    const bool up = y < 0.35f;
    const bool down = y > 0.65f;
    if (left && up) return GazeDirection::LeftUp;
    if (right && up) return GazeDirection::RightUp;
    if (left && down) return GazeDirection::LeftDown;
    if (right && down) return GazeDirection::RightDown;
    if (left) return GazeDirection::Left;
    if (right) return GazeDirection::Right;
    if (up) return GazeDirection::Up;
    if (down) return GazeDirection::Down;
    return GazeDirection::Center;
}

}  // namespace

class AttentionTask::Impl {
public:
    bool Initialize(const AttentionConfig& config) {
        config_ = config;
        filter_x_.reset(new OneEuroFilter(config.one_euro_min_cutoff,
                                          config.one_euro_beta,
                                          config.one_euro_derivative_cutoff));
        filter_y_.reset(new OneEuroFilter(config.one_euro_min_cutoff,
                                          config.one_euro_beta,
                                          config.one_euro_derivative_cutoff));
        return true;
    }

    AttentionState Process(const TrackingState& tracking) {
        AttentionState state;
        state.gaze = last_gaze_;
        state.calibration = calibrator_.State();
        if (!tracking.face.valid) {
            ResetFilter();
            return state;
        }

        const float face_width = tracking.face.box[2] - tracking.face.box[0];
        const float face_height = tracking.face.box[3] - tracking.face.box[1];
        if (face_width <= 1.f || face_height <= 1.f) return state;

        const bool left_valid = tracking.left_pupil.valid;
        const bool right_valid = tracking.right_pupil.valid;
        const float left_x = left_valid
            ? (tracking.left_pupil.position[0] - tracking.left_eye_center[0]) /
              face_width : 0.f;
        const float left_y = left_valid
            ? (tracking.left_pupil.position[1] - tracking.left_eye_center[1]) /
              face_height : 0.f;
        const float right_x = right_valid
            ? (tracking.right_pupil.position[0] - tracking.right_eye_center[0]) /
              face_width : 0.f;
        const float right_y = right_valid
            ? (tracking.right_pupil.position[1] - tracking.right_eye_center[1]) /
              face_height : 0.f;

        float raw_x = 0.f;
        float raw_y = 0.f;
        float confidence = 0.f;
        bool has_gaze = false;
        if (left_valid && right_valid) {
            const float left_confidence = tracking.left_pupil.confidence;
            const float right_confidence = tracking.right_pupil.confidence;
            const float sum = left_confidence + right_confidence;
            const float dx = left_x - right_x;
            const float dy = left_y - right_y;
            const float divergence = std::sqrt(dx * dx + dy * dy);
            if (divergence > 0.12f &&
                left_confidence > right_confidence * 1.20f) {
                raw_x = left_x;
                raw_y = left_y;
                confidence = left_confidence * 0.80f;
            } else if (divergence > 0.12f &&
                       right_confidence > left_confidence * 1.20f) {
                raw_x = right_x;
                raw_y = right_y;
                confidence = right_confidence * 0.80f;
            } else {
                raw_x = (left_confidence * left_x + right_confidence * right_x) /
                        std::max(0.001f, sum);
                raw_y = (left_confidence * left_y + right_confidence * right_y) /
                        std::max(0.001f, sum);
                confidence = 0.5f * sum * (divergence > 0.12f ? 0.55f : 1.f);
            }
            has_gaze = true;
        } else if (left_valid) {
            raw_x = left_x;
            raw_y = left_y;
            confidence = tracking.left_pupil.confidence * 0.75f;
            has_gaze = true;
        } else if (right_valid) {
            raw_x = right_x;
            raw_y = right_y;
            confidence = tracking.right_pupil.confidence * 0.75f;
            has_gaze = true;
        }

        if (has_gaze && confidence >= 0.30f && calibrator_.calibrating() &&
            left_valid && right_valid &&
            confidence >= config_.pupil_confidence_min) {
            calibrator_.AddSample(raw_x, raw_y);
        }

        if (has_gaze && confidence >= 0.30f) {
            const Point2f mapped = calibrator_.Map(raw_x, raw_y);
            last_gaze_[0] = Clamp01(filter_x_->Filter(mapped[0], tracking.timestamp));
            last_gaze_[1] = Clamp01(filter_y_->Filter(mapped[1], tracking.timestamp));
            filter_initialized_ = true;
            hold_frames_ = 0;
            state.gaze_valid = true;
            state.confidence = confidence;
        } else if (filter_initialized_ && hold_frames_ < 2) {
            hold_frames_++;
            state.gaze_valid = true;
            state.confidence = 0.f;
        } else {
            state.calibration = calibrator_.State();
            return state;
        }

        state.gaze = last_gaze_;
        state.direction = ClassifyGaze(last_gaze_[0], last_gaze_[1]);
        state.attentive = state.gaze_valid &&
                          state.direction == GazeDirection::Center;
        state.calibration = calibrator_.State();
        return state;
    }

    void StartCalibration() {
        calibrator_.Start();
        ResetFilter();
    }

    void Reset() {
        calibrator_.Reset();
        last_gaze_ = Point2f{{0.5f, 0.5f}};
        ResetFilter();
    }

private:
    void ResetFilter() {
        if (filter_x_) filter_x_->Reset();
        if (filter_y_) filter_y_->Reset();
        filter_initialized_ = false;
        hold_frames_ = 0;
    }

    AttentionConfig config_;
    GazeCalibrator calibrator_;
    std::unique_ptr<OneEuroFilter> filter_x_;
    std::unique_ptr<OneEuroFilter> filter_y_;
    Point2f last_gaze_{{0.5f, 0.5f}};
    int hold_frames_ = 0;
    bool filter_initialized_ = false;
};

AttentionTask::AttentionTask() : impl_(new Impl()) {}
AttentionTask::~AttentionTask() = default;

bool AttentionTask::Initialize(const AttentionConfig& config) {
    return impl_->Initialize(config);
}

AttentionState AttentionTask::Process(const TrackingState& tracking) {
    return impl_->Process(tracking);
}

void AttentionTask::StartCalibration() { impl_->StartCalibration(); }
void AttentionTask::Reset() { impl_->Reset(); }

}  // namespace eye_track
