#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <string>

#include "../include/common.hpp"

namespace eye_track {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;
using Point2f = std::array<float, 2>;
using RectF = std::array<float, 4>;

enum class PupilMode {
    Classic,
    Hybrid,
    Model
};

enum class TrackingMode {
    Reacquire = 0,
    Degraded = 1,
    Stable = 2
};

enum class GazeDirection {
    Center = 0,
    Left,
    Right,
    Up,
    Down,
    LeftUp,
    RightUp,
    LeftDown,
    RightDown
};

enum class TaskCommand {
    StartCalibration,
    Reset,
    ClearMarks
};

struct FramePacket {
    ssne_tensor_t* image = nullptr;
    uint64_t frame_id = 0;
    TimePoint captured_at;
};

struct FaceState {
    RectF box{{0.f, 0.f, 0.f, 0.f}};
    float confidence = 0.f;
    TrackingMode mode = TrackingMode::Reacquire;
    bool detector_ran = false;
    bool valid = false;
};

struct PupilState {
    Point2f position{{0.f, 0.f}};
    Point2f velocity{{0.f, 0.f}};
    float confidence = 0.f;
    float dark_ratio = 0.f;
    int missed_frames = 0;
    bool blink_measurement_valid = false;
    bool used_model = false;
    bool valid = false;
};

struct BlinkState {
    int count = 0;
    int duration_ms = 0;
    bool closed = false;
    bool event = false;
};

struct TrackingState {
    uint64_t frame_id = 0;
    TimePoint timestamp;
    FaceState face;
    RectF left_eye_box{{0.f, 0.f, 0.f, 0.f}};
    RectF right_eye_box{{0.f, 0.f, 0.f, 0.f}};
    Point2f left_eye_center{{0.f, 0.f}};
    Point2f right_eye_center{{0.f, 0.f}};
    PupilState left_pupil;
    PupilState right_pupil;
    BlinkState blink;
};

struct CalibrationState {
    Point2f target{{0.5f, 0.5f}};
    int point_index = 0;
    int sample_count = 0;
    bool active = false;
    bool calibrated = false;
};

struct AttentionState {
    Point2f gaze{{0.5f, 0.5f}};
    float confidence = 0.f;
    GazeDirection direction = GazeDirection::Center;
    CalibrationState calibration;
    bool gaze_valid = false;
    bool attentive = false;
};

struct EyeTrackingResult {
    TrackingState tracking;
    AttentionState attention;
};

struct TrackingConfig {
    int image_width = 640;
    int image_height = 480;
    float scrfd_tracking_hz = 30.f;
    float scrfd_degraded_hz = 60.f;
    float pupil_confidence_min = 0.45f;
    PupilMode pupil_mode = PupilMode::Hybrid;
    std::string face_model = "/app_demo/app_assets/models/face_640x480.m1model";
    std::string pupil_model = "/app_demo/app_assets/models/pupil_gap.m1model";
};

struct AttentionConfig {
    float pupil_confidence_min = 0.45f;
    float one_euro_min_cutoff = 3.f;
    float one_euro_beta = 0.6f;
    float one_euro_derivative_cutoff = 1.f;
};

struct TaskConfig {
    TrackingConfig tracking;
    AttentionConfig attention;
};

inline const char* TrackingModeName(TrackingMode mode) {
    switch (mode) {
        case TrackingMode::Stable: return "TRACKING";
        case TrackingMode::Degraded: return "DEGRADED";
        default: return "REACQUIRE";
    }
}

}  // namespace eye_track
