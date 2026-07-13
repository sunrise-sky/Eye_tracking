#include "tracking_task.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>

namespace eye_track {
namespace {

float Clamp01(float value) {
    return std::max(0.f, std::min(1.f, value));
}

struct PupilObservation {
    bool valid = false;
    float x = 0.f;
    float y = 0.f;
    float confidence = 0.f;
    float blink_dark_ratio = 0.f;
    bool blink_valid = false;
    bool used_model = false;
};

class PupilDetector {
public:
    bool Initialize(int width, int height, const std::string& model_path,
                    bool enable_model, bool prefer_model) {
        image_width_ = width;
        image_height_ = height;
        prefer_model_ = prefer_model;
        linux_tensor_ = create_tensor(width, height, SSNE_Y_8, SSNE_BUF_LINUX);
        linux_tensor_ready_ = true;
        image_buffer_.resize(static_cast<size_t>(width) * height);
        printf("[PupilDetector] classic mode ready\n");
        if (enable_model) InitializeModel(model_path);
        return true;
    }

    bool PrepareFrame(ssne_tensor_t* image) {
        if (!image || image_buffer_.empty()) return false;
        if (copy_tensor_buffer(*image, linux_tensor_) != SSNE_ERRCODE_NO_ERROR)
            return false;
        return save_tensor_buffer_ptr(linux_tensor_, image_buffer_.data(),
                                      static_cast<int>(image_buffer_.size())) ==
               SSNE_ERRCODE_NO_ERROR;
    }

    PupilObservation Detect(const RectF& eye_box,
                            const PupilObservation* previous,
                            bool allow_model_recovery) {
        int x1, y1, x2, y2;
        if (!ClipEyeBox(eye_box, x1, y1, x2, y2)) return PupilObservation();
        const int roi_width = x2 - x1;
        const int roi_height = y2 - y1;

        if (prefer_model_ && allow_model_recovery && model_ready_) {
            PupilObservation model = DetectWithModel(
                x1, y1, roi_width, roi_height, previous);
            if (model.valid) {
                model.blink_dark_ratio = ComputeBlinkDarkRatio(
                    x1, y1, roi_width, roi_height);
                model.blink_valid = true;
                return model;
            }
        }

        PupilObservation classic = DetectClassic(
            x1, y1, roi_width, roi_height, previous);
        if (allow_model_recovery && model_ready_ &&
            (!classic.valid || classic.confidence < kModelFallbackConfidence)) {
            PupilObservation model = DetectWithModel(
                x1, y1, roi_width, roi_height, previous);
            if (model.valid) {
                model.blink_dark_ratio = classic.blink_dark_ratio;
                model.blink_valid = classic.blink_valid;
                return model;
            }
        }
        return classic;
    }

    bool PreferModel() const { return prefer_model_; }

    const char* ModeName() const {
        if (!model_ready_) return "classic";
        return prefer_model_ ? "model primary + classic fallback"
                             : "classic + model recovery";
    }

    void Release() {
        if (output_ready_) {
            release_tensor(outputs_[0]);
            output_ready_ = false;
        }
        if (model_input_ready_) {
            release_tensor(model_input_);
            model_input_ready_ = false;
        }
        if (linux_tensor_ready_) {
            release_tensor(linux_tensor_);
            linux_tensor_ready_ = false;
        }
        image_buffer_.clear();
        model_ready_ = false;
    }

private:
    static const int kModelInputSize = 224;
    static const int kHistogramBins = 64;
    static constexpr float kModelFallbackConfidence = 0.45f;

    PupilObservation DetectClassic(int eye_x, int eye_y, int roi_width,
                                   int roi_height,
                                   const PupilObservation* previous) const {
        PupilObservation result;
        int histogram[kHistogramBins] = {0};
        uint64_t intensity_sum = 0;
        const int roi_pixels = roi_width * roi_height;
        if (roi_pixels <= 0) return result;

        for (int y = eye_y; y < eye_y + roi_height; ++y) {
            const uint8_t* row = image_buffer_.data() + y * image_width_;
            for (int x = eye_x; x < eye_x + roi_width; ++x) {
                const uint8_t pixel = row[x];
                histogram[pixel >> 2]++;
                intensity_sum += pixel;
            }
        }

        const float mean = static_cast<float>(intensity_sum) / roi_pixels;
        const int blink_bin = std::max(0, std::min(
            kHistogramBins - 1, static_cast<int>(mean * 0.6f) >> 2));
        int blink_dark = 0;
        for (int i = 0; i <= blink_bin; ++i) blink_dark += histogram[i];
        result.blink_dark_ratio = static_cast<float>(blink_dark) / roi_pixels;
        result.blink_valid = true;

        const int percentile_target = std::max(4, roi_pixels * 18 / 100);
        int cumulative = 0;
        int percentile_bin = 0;
        for (; percentile_bin < kHistogramBins; ++percentile_bin) {
            cumulative += histogram[percentile_bin];
            if (cumulative >= percentile_target) break;
        }
        const int percentile_threshold = std::min(255, percentile_bin * 4 + 3);
        const int pupil_threshold = std::max(2, std::min(
            percentile_threshold, static_cast<int>(mean * 0.78f)));

        int sx1 = eye_x;
        int sy1 = eye_y;
        int sx2 = eye_x + roi_width;
        int sy2 = eye_y + roi_height;
        if (previous && previous->valid && previous->confidence >= 0.40f &&
            previous->x >= eye_x && previous->x < eye_x + roi_width &&
            previous->y >= eye_y && previous->y < eye_y + roi_height) {
            const int half_width = std::max(6, static_cast<int>(roi_width * 0.38f));
            const int half_height = std::max(4, static_cast<int>(roi_height * 0.42f));
            const int px = static_cast<int>(previous->x + 0.5f);
            const int py = static_cast<int>(previous->y + 0.5f);
            sx1 = std::max(eye_x, px - half_width);
            sy1 = std::max(eye_y, py - half_height);
            sx2 = std::min(eye_x + roi_width, px + half_width + 1);
            sy2 = std::min(eye_y + roi_height, py + half_height + 1);
        }

        double weight_sum = 0.0;
        double weighted_x = 0.0;
        double weighted_y = 0.0;
        double weighted_x2 = 0.0;
        double weighted_y2 = 0.0;
        uint64_t dark_intensity_sum = 0;
        int dark_count = 0;
        for (int y = sy1; y < sy2; ++y) {
            const uint8_t* row = image_buffer_.data() + y * image_width_;
            for (int x = sx1; x < sx2; ++x) {
                const int pixel = row[x];
                if (pixel > pupil_threshold) continue;
                const double weight = pupil_threshold - pixel + 1;
                weight_sum += weight;
                weighted_x += weight * x;
                weighted_y += weight * y;
                weighted_x2 += weight * x * x;
                weighted_y2 += weight * y * y;
                dark_intensity_sum += pixel;
                dark_count++;
            }
        }

        const int search_pixels = std::max(1, (sx2 - sx1) * (sy2 - sy1));
        if (dark_count < 4 || weight_sum <= 0.0) return result;

        const float cx = static_cast<float>(weighted_x / weight_sum);
        const float cy = static_cast<float>(weighted_y / weight_sum);
        const float variance_x = std::max(0.f, static_cast<float>(
            weighted_x2 / weight_sum - cx * cx));
        const float variance_y = std::max(0.f, static_cast<float>(
            weighted_y2 / weight_sum - cy * cy));
        const float spread = std::sqrt(variance_x) / std::max(1, roi_width) +
                             std::sqrt(variance_y) / std::max(1, roi_height);
        const float compactness = Clamp01((0.34f - spread) / 0.25f);
        const float dark_mean = static_cast<float>(dark_intensity_sum) / dark_count;
        const float contrast = Clamp01(
            (mean - dark_mean) / std::max(1.f, mean) * 2.5f);
        const float dark_ratio = static_cast<float>(dark_count) / search_pixels;
        const float ratio_score = dark_ratio <= 0.12f
            ? Clamp01(dark_ratio / 0.12f)
            : Clamp01((0.48f - dark_ratio) / 0.36f);

        float temporal_score = 0.65f;
        if (previous && previous->valid) {
            const float dx = cx - previous->x;
            const float dy = cy - previous->y;
            const float distance = std::sqrt(dx * dx + dy * dy);
            const float radius = std::max(4.f, roi_width * 0.28f);
            const float normalized = distance / radius;
            temporal_score = 1.f / (1.f + normalized * normalized);
        }
        const float edge_x = std::min(cx - eye_x, eye_x + roi_width - 1 - cx);
        const float edge_y = std::min(cy - eye_y, eye_y + roi_height - 1 - cy);
        const float edge_score = Clamp01(std::min(
            edge_x / std::max(1.f, roi_width * 0.08f),
            edge_y / std::max(1.f, roi_height * 0.08f)));

        result.x = cx;
        result.y = cy;
        result.confidence = Clamp01(0.35f * contrast + 0.20f * ratio_score +
                                    0.20f * compactness +
                                    0.15f * temporal_score +
                                    0.10f * edge_score);
        result.valid = dark_ratio < 0.48f && result.confidence >= 0.28f;
        return result;
    }

    void InitializeModel(const std::string& model_path) {
        if (model_path.empty()) return;
        std::ifstream input(model_path.c_str(), std::ios::binary);
        if (!input.good()) {
            printf("[PupilDetector] optional model not found: %s, use classic\n",
                   model_path.c_str());
            return;
        }

        char* path = const_cast<char*>(model_path.c_str());
        model_id_ = ssne_loadmodel(path, SSNE_STATIC_ALLOC);
        if (ssne_get_model_input_num(model_id_) != 1) {
            printf("[PupilDetector] model input count invalid, use classic\n");
            return;
        }

        int data_type = SSNE_UINT8;
        if (ssne_get_model_input_dtype(model_id_, &data_type) !=
            SSNE_ERRCODE_NO_ERROR) {
            printf("[PupilDetector] model dtype query failed, use classic\n");
            return;
        }
        if (data_type != SSNE_UINT8 && data_type != SSNE_FLOAT32) {
            printf("[PupilDetector] unsupported model dtype=%d, use classic\n",
                   data_type);
            return;
        }

        model_input_type_ = data_type;
        if (data_type == SSNE_FLOAT32) {
            model_input_ = create_tensor(kModelInputSize, kModelInputSize,
                                         SSNE_BYTES, SSNE_BUF_AI);
            set_data_type(model_input_, SSNE_FLOAT32);
            model_input_f32_.resize(kModelInputSize * kModelInputSize);
        } else {
            model_input_ = create_tensor(kModelInputSize, kModelInputSize,
                                         SSNE_Y_8, SSNE_BUF_AI);
            model_input_u8_.resize(kModelInputSize * kModelInputSize);
        }
        model_input_ready_ = true;
        model_ready_ = true;
        printf("[PupilDetector] optional model ready: %s dtype=%d\n",
               model_path.c_str(), model_input_type_);
    }

    PupilObservation DetectWithModel(int eye_x, int eye_y, int roi_width,
                                     int roi_height,
                                     const PupilObservation* previous) {
        PupilObservation result;
        if (!FillModelInput(eye_x, eye_y, roi_width, roi_height)) {
            DisableModel("[PupilDetector] model input failed, switch to classic\n");
            return result;
        }
        if (ssne_inference(model_id_, 1, &model_input_) !=
            SSNE_ERRCODE_NO_ERROR) {
            DisableModel("[PupilDetector] model inference failed, switch to classic\n");
            return result;
        }
        if (ssne_getoutput(model_id_, 1, outputs_) !=
            SSNE_ERRCODE_NO_ERROR) {
            DisableModel("[PupilDetector] model output failed, switch to classic\n");
            return result;
        }
        output_ready_ = true;
        if (get_data_type(outputs_[0]) != SSNE_FLOAT32 ||
            get_mem_size(outputs_[0]) < 2 * sizeof(float)) {
            DisableModel("[PupilDetector] model output type invalid, switch to classic\n");
            return result;
        }

        float* output = static_cast<float*>(get_data(outputs_[0]));
        if (!output || !std::isfinite(output[0]) || !std::isfinite(output[1])) {
            DisableModel("[PupilDetector] model output data invalid, switch to classic\n");
            return result;
        }
        if (output[0] < -0.25f || output[0] > 1.25f ||
            output[1] < -0.25f || output[1] > 1.25f) {
            DisableModel("[PupilDetector] model output range invalid, switch to classic\n");
            return result;
        }

        const float nx = Clamp01(output[0]);
        const float ny = Clamp01(output[1]);
        result.x = eye_x + nx * roi_width;
        result.y = eye_y + ny * roi_height;
        result.confidence = 0.78f;
        if (previous && previous->valid) {
            const float dx = result.x - previous->x;
            const float dy = result.y - previous->y;
            const float normalized = std::sqrt(dx * dx + dy * dy) /
                                     std::max(4.f, roi_width * 0.35f);
            result.confidence *= 1.f / (1.f + normalized * normalized);
        }
        result.valid = result.confidence >= 0.35f;
        result.used_model = true;
        return result;
    }

    float ComputeBlinkDarkRatio(int eye_x, int eye_y, int roi_width,
                                int roi_height) const {
        const int pixels = roi_width * roi_height;
        if (pixels <= 0) return 0.f;
        uint64_t sum = 0;
        for (int y = eye_y; y < eye_y + roi_height; ++y) {
            const uint8_t* row = image_buffer_.data() + y * image_width_;
            for (int x = eye_x; x < eye_x + roi_width; ++x) sum += row[x];
        }
        const float threshold = static_cast<float>(sum) / pixels * 0.6f;
        int dark = 0;
        for (int y = eye_y; y < eye_y + roi_height; ++y) {
            const uint8_t* row = image_buffer_.data() + y * image_width_;
            for (int x = eye_x; x < eye_x + roi_width; ++x)
                if (row[x] < threshold) dark++;
        }
        return static_cast<float>(dark) / pixels;
    }

    bool FillModelInput(int eye_x, int eye_y, int roi_width, int roi_height) {
        const int pixels = kModelInputSize * kModelInputSize;
        for (int y = 0; y < kModelInputSize; ++y) {
            float source_y = eye_y +
                (y + 0.5f) * roi_height / kModelInputSize - 0.5f;
            source_y = std::max(static_cast<float>(eye_y), std::min(
                static_cast<float>(eye_y + roi_height - 1), source_y));
            const int y0 = static_cast<int>(std::floor(source_y));
            const int y1 = std::min(eye_y + roi_height - 1, y0 + 1);
            const float fy = source_y - y0;

            for (int x = 0; x < kModelInputSize; ++x) {
                float source_x = eye_x +
                    (x + 0.5f) * roi_width / kModelInputSize - 0.5f;
                source_x = std::max(static_cast<float>(eye_x), std::min(
                    static_cast<float>(eye_x + roi_width - 1), source_x));
                const int x0 = static_cast<int>(std::floor(source_x));
                const int x1 = std::min(eye_x + roi_width - 1, x0 + 1);
                const float fx = source_x - x0;
                const float p00 = image_buffer_[y0 * image_width_ + x0];
                const float p01 = image_buffer_[y0 * image_width_ + x1];
                const float p10 = image_buffer_[y1 * image_width_ + x0];
                const float p11 = image_buffer_[y1 * image_width_ + x1];
                const float top = p00 + (p01 - p00) * fx;
                const float bottom = p10 + (p11 - p10) * fx;
                const float pixel = top + (bottom - top) * fy;
                const int index = y * kModelInputSize + x;
                if (model_input_type_ == SSNE_FLOAT32)
                    model_input_f32_[index] = pixel / 255.f;
                else
                    model_input_u8_[index] = static_cast<uint8_t>(
                        std::max(0, std::min(255, static_cast<int>(pixel + 0.5f))));
            }
        }

        if (model_input_type_ == SSNE_FLOAT32) {
            return load_tensor_buffer_ptr(
                model_input_, model_input_f32_.data(),
                pixels * static_cast<int>(sizeof(float))) ==
                SSNE_ERRCODE_NO_ERROR;
        }
        return load_tensor_buffer_ptr(model_input_, model_input_u8_.data(),
                                      pixels) == SSNE_ERRCODE_NO_ERROR;
    }

    void DisableModel(const char* message) {
        if (model_ready_) printf("%s", message);
        model_ready_ = false;
    }

    bool ClipEyeBox(const RectF& box, int& x1, int& y1, int& x2, int& y2) const {
        x1 = std::max(0, std::min(image_width_,
            static_cast<int>(std::floor(box[0]))));
        y1 = std::max(0, std::min(image_height_,
            static_cast<int>(std::floor(box[1]))));
        x2 = std::max(0, std::min(image_width_,
            static_cast<int>(std::ceil(box[2]))));
        y2 = std::max(0, std::min(image_height_,
            static_cast<int>(std::ceil(box[3]))));
        return x2 - x1 >= 4 && y2 - y1 >= 4;
    }

    int image_width_ = 0;
    int image_height_ = 0;
    ssne_tensor_t linux_tensor_;
    bool linux_tensor_ready_ = false;
    std::vector<uint8_t> image_buffer_;
    bool prefer_model_ = false;
    bool model_ready_ = false;
    bool model_input_ready_ = false;
    bool output_ready_ = false;
    int model_input_type_ = SSNE_UINT8;
    uint16_t model_id_ = 0;
    ssne_tensor_t model_input_;
    ssne_tensor_t outputs_[1];
    std::vector<uint8_t> model_input_u8_;
    std::vector<float> model_input_f32_;
};

constexpr float PupilDetector::kModelFallbackConfidence;

class FaceTracker {
public:
    FaceTracker(int width, int height, const TrackingConfig& config)
        : width_(width), height_(height), config_(config) {
        Reset();
    }

    void Reset() {
        valid_ = false;
        mode_ = TrackingMode::Reacquire;
        box_.fill(0.f);
        velocity_.fill(0.f);
        missed_detections_ = 0;
        low_confidence_streak_ = 0;
        stable_pupil_streak_ = 0;
        detector_attempted_ = false;
        detector_succeeded_ = false;
        time_initialized_ = false;
    }

    bool ShouldRunDetector(TimePoint now) const {
        if (!valid_ || mode_ == TrackingMode::Reacquire ||
            !detector_attempted_) return true;
        const float hz = mode_ == TrackingMode::Stable
            ? config_.scrfd_tracking_hz : config_.scrfd_degraded_hz;
        return std::chrono::duration<float>(now - last_detector_attempt_).count()
               >= 1.f / std::max(1.f, hz);
    }

    void Predict(TimePoint now) {
        if (!valid_ || !time_initialized_) {
            state_time_ = now;
            time_initialized_ = true;
            return;
        }
        float dt = std::chrono::duration<float>(now - state_time_).count();
        dt = std::max(0.f, std::min(0.1f, dt));
        for (size_t i = 0; i < box_.size(); ++i) box_[i] += velocity_[i] * dt;
        ClampBox(box_);
        state_time_ = now;
    }

    void UpdateDetection(const RectF& measurement, TimePoint now) {
        last_detector_attempt_ = now;
        detector_attempted_ = true;
        RectF clipped = measurement;
        ClampBox(clipped);
        if (!valid_) {
            box_ = clipped;
            velocity_.fill(0.f);
        } else {
            float dt = detector_succeeded_
                ? std::chrono::duration<float>(now - last_detector_success_).count()
                : 1.f / config_.scrfd_tracking_hz;
            dt = std::max(1.f / 180.f, std::min(0.25f, dt));
            for (size_t i = 0; i < box_.size(); ++i) {
                const float residual = clipped[i] - box_[i];
                box_[i] += 0.70f * residual;
                velocity_[i] += 0.05f * residual / dt;
            }
            ClampBox(box_);
        }
        valid_ = true;
        detector_succeeded_ = true;
        last_detector_success_ = now;
        missed_detections_ = 0;
        if (mode_ == TrackingMode::Reacquire) mode_ = TrackingMode::Degraded;
        state_time_ = now;
        time_initialized_ = true;
    }

    void UpdateDetectionMiss(TimePoint now) {
        last_detector_attempt_ = now;
        detector_attempted_ = true;
        missed_detections_++;
        const float age = detector_succeeded_
            ? std::chrono::duration<float>(now - last_detector_success_).count()
            : 1.f;
        if (!valid_ || missed_detections_ >= 3 || age > 0.15f) {
            valid_ = false;
            mode_ = TrackingMode::Reacquire;
            velocity_.fill(0.f);
        } else {
            mode_ = TrackingMode::Degraded;
        }
    }

    void UpdatePupilQuality(const PupilObservation& left,
                            const PupilObservation& right) {
        const bool any_valid = left.valid || right.valid;
        const bool both_valid = left.valid && right.valid;
        const float confidence = both_valid
            ? 0.5f * (left.confidence + right.confidence)
            : (left.valid ? left.confidence : right.confidence) * 0.75f;
        if (!any_valid || confidence < config_.pupil_confidence_min) {
            low_confidence_streak_++;
            stable_pupil_streak_ = 0;
            mode_ = low_confidence_streak_ >= 3
                ? TrackingMode::Reacquire : TrackingMode::Degraded;
            if (low_confidence_streak_ >= 3) valid_ = false;
            return;
        }
        low_confidence_streak_ = 0;
        stable_pupil_streak_++;
        if (stable_pupil_streak_ >= 6 && missed_detections_ == 0)
            mode_ = TrackingMode::Stable;
        else if (mode_ == TrackingMode::Reacquire)
            mode_ = TrackingMode::Degraded;
    }

    bool Valid() const { return valid_; }
    TrackingMode Mode() const { return mode_; }
    const RectF& Box() const { return box_; }

private:
    void ClampBox(RectF& box) const {
        box[0] = std::max(0.f, std::min(static_cast<float>(width_ - 2), box[0]));
        box[1] = std::max(0.f, std::min(static_cast<float>(height_ - 2), box[1]));
        box[2] = std::max(box[0] + 2.f,
                          std::min(static_cast<float>(width_), box[2]));
        box[3] = std::max(box[1] + 2.f,
                          std::min(static_cast<float>(height_), box[3]));
    }

    int width_;
    int height_;
    TrackingConfig config_;
    bool valid_ = false;
    TrackingMode mode_ = TrackingMode::Reacquire;
    RectF box_;
    RectF velocity_;
    int missed_detections_ = 0;
    int low_confidence_streak_ = 0;
    int stable_pupil_streak_ = 0;
    bool detector_attempted_ = false;
    bool detector_succeeded_ = false;
    bool time_initialized_ = false;
    TimePoint state_time_;
    TimePoint last_detector_attempt_;
    TimePoint last_detector_success_;
};

class BlinkDetector {
public:
    BlinkState Update(bool valid, float dark_ratio) {
        BlinkState result;
        result.count = blink_count_;
        result.closed = blink_active_;
        if (!valid) {
            closed_streak_ = 0;
            open_streak_ = 0;
            if (!suppress_until_open_) blink_active_ = false;
            return result;
        }

        if (!baseline_ready_) {
            open_baseline_ = std::max(0.02f, dark_ratio);
            baseline_ready_ = true;
        }
        const float close_threshold = std::max(0.015f, open_baseline_ * 0.60f);
        const float open_threshold = std::max(close_threshold + 0.008f,
                                              open_baseline_ * 0.78f);
        const bool is_closed = dark_ratio < close_threshold;
        const bool is_open = dark_ratio > open_threshold;
        const TimePoint now = Clock::now();

        if (!blink_active_ && !suppress_until_open_ && !is_closed) {
            open_baseline_ = 0.99f * open_baseline_ + 0.01f * dark_ratio;
            open_baseline_ = std::max(0.02f, std::min(0.40f, open_baseline_));
        }
        if (suppress_until_open_) {
            if (is_open && ++open_streak_ >= 2) {
                suppress_until_open_ = false;
                open_streak_ = 0;
            } else if (!is_open) {
                open_streak_ = 0;
            }
            return result;
        }
        if (!blink_active_) {
            if (is_closed) {
                if (closed_streak_ == 0) blink_start_ = now;
                if (++closed_streak_ >= 2) {
                    blink_active_ = true;
                    open_streak_ = 0;
                }
            } else {
                closed_streak_ = 0;
            }
            result.closed = blink_active_;
            return result;
        }
        if (is_open) {
            if (open_streak_ == 0) open_start_ = now;
            if (++open_streak_ < 2) return result;
            const int duration = static_cast<int>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    open_start_ - blink_start_).count());
            if (duration >= 80 && duration <= 800) {
                blink_count_++;
                result.event = true;
                result.duration_ms = duration;
                printf("[Blink] count=%d duration_ms=%d\n", blink_count_, duration);
            }
            blink_active_ = false;
            closed_streak_ = 0;
            open_streak_ = 0;
            result.count = blink_count_;
            result.closed = false;
            return result;
        }
        open_streak_ = 0;
        const int duration = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now - blink_start_).count());
        if (duration > 800) {
            blink_active_ = false;
            closed_streak_ = 0;
            suppress_until_open_ = true;
        }
        result.closed = blink_active_;
        return result;
    }

    void Reset() { *this = BlinkDetector(); }

private:
    int closed_streak_ = 0;
    int open_streak_ = 0;
    int blink_count_ = 0;
    bool blink_active_ = false;
    bool suppress_until_open_ = false;
    bool baseline_ready_ = false;
    float open_baseline_ = 0.f;
    TimePoint blink_start_;
    TimePoint open_start_;
};

RectF GetEyeBox(const RectF& face, float center_x, float center_y,
                float image_width, float image_height) {
    const float face_width = face[2] - face[0];
    const float half_width = face_width * 0.20f;
    const float half_height = half_width * 0.6f;
    float x1 = center_x - half_width;
    float y1 = center_y - half_height;
    float x2 = center_x + half_width;
    float y2 = center_y + half_height;
    if (x1 < 0.f) { x2 -= x1; x1 = 0.f; }
    if (y1 < 0.f) { y2 -= y1; y1 = 0.f; }
    if (x2 > image_width) { x1 -= x2 - image_width; x2 = image_width; }
    if (y2 > image_height) { y1 -= y2 - image_height; y2 = image_height; }
    return RectF{{std::max(0.f, x1), std::max(0.f, y1), x2, y2}};
}

PupilState PublishPupil(const PupilObservation& observation,
                        const PupilState& previous, int missed_frames,
                        TimePoint now, TimePoint previous_time) {
    PupilState state;
    state.position = Point2f{{observation.x, observation.y}};
    state.confidence = observation.confidence;
    state.dark_ratio = observation.blink_dark_ratio;
    state.blink_measurement_valid = observation.blink_valid;
    state.used_model = observation.used_model;
    state.valid = observation.valid;
    state.missed_frames = observation.valid ? 0 : missed_frames;
    if (observation.valid && previous.valid) {
        float dt = std::chrono::duration<float>(now - previous_time).count();
        dt = std::max(1.f / 180.f, std::min(0.25f, dt));
        state.velocity = Point2f{{
            (observation.x - previous.position[0]) / dt,
            (observation.y - previous.position[1]) / dt}};
    }
    return state;
}

}  // namespace

class TrackingTask::Impl {
public:
    bool Initialize(const TrackingConfig& config) {
        config_ = config;
        std::array<int, 2> image_shape{{config.image_width, config.image_height}};
        std::array<int, 2> detector_shape{{640, 480}};
        std::string face_model = config.face_model;
        const int box_length = detector_shape[0] * detector_shape[1] / 512 * 21;
        face_detector_.Initialize(face_model, &image_shape, &detector_shape,
                                  false, box_length);

        const bool enable_model = config.pupil_mode != PupilMode::Classic;
        const bool prefer_model = config.pupil_mode == PupilMode::Model;
        pupil_detector_.Initialize(config.image_width, config.image_height,
                                   config.pupil_model,
                                   enable_model, prefer_model);
        printf("[INFO] Pupil detector OK (%s)\n", pupil_detector_.ModeName());

        mirrored_ = create_tensor(config.image_width, config.image_height,
                                  SSNE_Y_8, SSNE_BUF_AI);
        mirrored_ready_ = true;
        face_tracker_.reset(new FaceTracker(config.image_width,
                                            config.image_height, config));
        initialized_ = true;
        return true;
    }

    TrackingState Process(const FramePacket& frame) {
        TrackingState state;
        state.frame_id = frame.frame_id;
        state.timestamp = Clock::now();
        if (!initialized_ || !frame.image) return state;

        mirror_tensor(*frame.image, mirrored_);
        const TimePoint now = Clock::now();
        state.timestamp = now;
        face_tracker_->Predict(now);

        FaceDetectionResult detection;
        const bool ran_detector = face_tracker_->ShouldRunDetector(now);
        if (ran_detector) {
            face_detector_.Predict(&mirrored_, &detection, 0.4f);
            detector_runs_++;
            if (!detection.boxes.empty()) {
                face_tracker_->UpdateDetection(detection.boxes[0], now);
                face_confidence_ = detection.scores.empty() ? 1.f : detection.scores[0];
            } else {
                face_tracker_->UpdateDetectionMiss(now);
                face_confidence_ *= 0.7f;
            }
        }

        state.face.detector_ran = ran_detector;
        state.face.mode = face_tracker_->Mode();
        state.face.valid = face_tracker_->Valid();
        state.face.confidence = face_confidence_;
        if (!state.face.valid) {
            state.blink = blink_detector_.Update(false, 0.f);
            left_history_ = PupilObservation();
            right_history_ = PupilObservation();
            return state;
        }

        state.face.box = face_tracker_->Box();
        const float face_width = state.face.box[2] - state.face.box[0];
        const float face_height = state.face.box[3] - state.face.box[1];
        float left_x = state.face.box[0] + face_width * 0.30f;
        float left_y = state.face.box[1] + face_height * 0.35f;
        float right_x = state.face.box[0] + face_width * 0.70f;
        float right_y = state.face.box[1] + face_height * 0.35f;
        if (ran_detector && detection.landmarks_per_face >= 2 &&
            detection.landmarks.size() >= 2) {
            left_x = detection.landmarks[0][0];
            left_y = detection.landmarks[0][1];
            right_x = detection.landmarks[1][0];
            right_y = detection.landmarks[1][1];
        }
        state.left_eye_center = Point2f{{left_x, left_y}};
        state.right_eye_center = Point2f{{right_x, right_y}};
        state.left_eye_box = GetEyeBox(state.face.box, left_x, left_y,
                                       config_.image_width, config_.image_height);
        state.right_eye_box = GetEyeBox(state.face.box, right_x, right_y,
                                        config_.image_width, config_.image_height);

        PupilObservation left;
        PupilObservation right;
        const bool frame_ready = pupil_detector_.PrepareFrame(&mirrored_);
        const bool allow_model = pupil_detector_.PreferModel() ||
            face_tracker_->Mode() != TrackingMode::Stable;
        if (frame_ready) {
            left = pupil_detector_.Detect(
                state.left_eye_box,
                left_history_.valid ? &left_history_ : nullptr,
                allow_model);
            right = pupil_detector_.Detect(
                state.right_eye_box,
                right_history_.valid ? &right_history_ : nullptr,
                allow_model);
        }
        if (left.used_model) model_recovery_runs_++;
        if (right.used_model) model_recovery_runs_++;

        if (left.valid) left_history_ = left;
        else {
            left_history_.confidence *= 0.70f;
            if (left_history_.confidence < 0.20f) left_history_.valid = false;
        }
        if (right.valid) right_history_ = right;
        else {
            right_history_.confidence *= 0.70f;
            if (right_history_.confidence < 0.20f) right_history_.valid = false;
        }

        face_tracker_->UpdatePupilQuality(left, right);
        state.face.mode = face_tracker_->Mode();

        if (!left.valid) left_missed_++; else left_missed_ = 0;
        if (!right.valid) right_missed_++; else right_missed_ = 0;
        state.left_pupil = PublishPupil(left, last_left_state_, left_missed_,
                                        now, previous_pupil_time_);
        state.right_pupil = PublishPupil(right, last_right_state_, right_missed_,
                                         now, previous_pupil_time_);
        if (state.left_pupil.valid) last_left_state_ = state.left_pupil;
        if (state.right_pupil.valid) last_right_state_ = state.right_pupil;
        previous_pupil_time_ = now;

        float blink_ratio = 0.f;
        bool blink_valid = false;
        if (left.blink_valid && right.blink_valid) {
            blink_ratio = 0.5f * (left.blink_dark_ratio + right.blink_dark_ratio);
            blink_valid = true;
        } else if (left.blink_valid) {
            blink_ratio = left.blink_dark_ratio;
            blink_valid = true;
        } else if (right.blink_valid) {
            blink_ratio = right.blink_dark_ratio;
            blink_valid = true;
        }
        state.blink = blink_detector_.Update(blink_valid, blink_ratio);
        return state;
    }

    void Reset() {
        if (face_tracker_) face_tracker_->Reset();
        left_history_ = PupilObservation();
        right_history_ = PupilObservation();
        last_left_state_ = PupilState();
        last_right_state_ = PupilState();
        left_missed_ = 0;
        right_missed_ = 0;
        face_confidence_ = 0.f;
        blink_detector_.Reset();
    }

    void Release() {
        if (!initialized_) return;
        pupil_detector_.Release();
        face_detector_.Release();
        if (mirrored_ready_) {
            release_tensor(mirrored_);
            mirrored_ready_ = false;
        }
        face_tracker_.reset();
        initialized_ = false;
    }

    uint64_t detector_runs() const { return detector_runs_; }
    uint64_t model_recovery_runs() const { return model_recovery_runs_; }

private:
    TrackingConfig config_;
    SCRFDGRAY face_detector_;
    PupilDetector pupil_detector_;
    std::unique_ptr<FaceTracker> face_tracker_;
    BlinkDetector blink_detector_;
    ssne_tensor_t mirrored_;
    bool mirrored_ready_ = false;
    bool initialized_ = false;
    float face_confidence_ = 0.f;
    PupilObservation left_history_;
    PupilObservation right_history_;
    PupilState last_left_state_;
    PupilState last_right_state_;
    int left_missed_ = 0;
    int right_missed_ = 0;
    TimePoint previous_pupil_time_ = Clock::now();
    uint64_t detector_runs_ = 0;
    uint64_t model_recovery_runs_ = 0;
};

TrackingTask::TrackingTask() : impl_(new Impl()) {}
TrackingTask::~TrackingTask() { impl_->Release(); }

bool TrackingTask::Initialize(const TrackingConfig& config) {
    return impl_->Initialize(config);
}

TrackingState TrackingTask::Process(const FramePacket& frame) {
    return impl_->Process(frame);
}

void TrackingTask::Reset() { impl_->Reset(); }
void TrackingTask::Release() { impl_->Release(); }

}  // namespace eye_track
