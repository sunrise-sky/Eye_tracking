#include "eye_tracking_app.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sys/select.h>
#include <unistd.h>
#endif

#include "../include/utils.hpp"
#include "../task/eye_tracking.hpp"

namespace eye_track {
namespace {

bool TensorHasCapacity(const ssne_tensor_t& tensor, size_t required_bytes) {
    return get_mem_size(tensor) >= required_bytes;
}

float ReadEnvFloat(const char* name, float default_value,
                   float minimum, float maximum) {
    const char* raw = std::getenv(name);
    if (!raw || !raw[0]) return default_value;
    char* end = nullptr;
    errno = 0;
    const float parsed = std::strtof(raw, &end);
    if (end == raw || *end != '\0' || errno == ERANGE ||
        !std::isfinite(parsed)) {
        fprintf(stderr, "[CONFIG] invalid %s=%s, use %.4g\n",
                name, raw, default_value);
        return default_value;
    }
    return std::max(minimum, std::min(maximum, parsed));
}

TaskConfig LoadTaskConfig(const std::string& default_mode) {
    TaskConfig config;
    config.tracking.scrfd_tracking_hz = ReadEnvFloat(
        "SCRFD_TRACKING_HZ", 30.f, 5.f, 120.f);
    config.tracking.scrfd_degraded_hz = ReadEnvFloat(
        "SCRFD_DEGRADED_HZ", 60.f, 10.f, 180.f);
    config.tracking.pupil_confidence_min = ReadEnvFloat(
        "PUPIL_CONFIDENCE_MIN", 0.45f, 0.1f, 0.9f);
    config.tracking.pupil_kalman_enabled =
        ReadEnvFloat("PUPIL_KALMAN_ENABLED", 1.f, 0.f, 1.f) > 0.5f;
    config.tracking.pupil_kalman_process_noise = ReadEnvFloat(
        "PUPIL_KALMAN_PROCESS_NOISE", 1800.f, 0.f, 12000.f);
    config.tracking.pupil_prediction_hold_frames = static_cast<int>(
        ReadEnvFloat("PUPIL_PREDICTION_HOLD_FRAMES", 3.f, 0.f, 12.f));
    config.attention.pupil_confidence_min =
        config.tracking.pupil_confidence_min;
    config.attention.one_euro_min_cutoff = ReadEnvFloat(
        "ONE_EURO_MIN_CUTOFF", 3.f, 0.1f, 20.f);
    config.attention.one_euro_beta = ReadEnvFloat(
        "ONE_EURO_BETA", 0.6f, 0.f, 5.f);
    config.attention.one_euro_derivative_cutoff = ReadEnvFloat(
        "ONE_EURO_D_CUTOFF", 1.f, 0.1f, 20.f);
    config.attention.gaze_kalman_enabled =
        ReadEnvFloat("GAZE_KALMAN_ENABLED", 1.f, 0.f, 1.f) > 0.5f;
    config.attention.gaze_kalman_process_noise = ReadEnvFloat(
        "GAZE_KALMAN_PROCESS_NOISE", 8.f, 0.f, 80.f);
    config.attention.gaze_kalman_measurement_noise = ReadEnvFloat(
        "GAZE_KALMAN_MEASUREMENT_NOISE", 0.0004f, 0.000001f, 0.02f);
    config.attention.gaze_prediction_hold_frames = static_cast<int>(
        ReadEnvFloat("GAZE_PREDICTION_HOLD_FRAMES", 4.f, 0.f, 12.f));

    const char* pupil_model = std::getenv("PUPIL_GAP_MODEL");
    if (pupil_model && pupil_model[0])
        config.tracking.pupil_model = pupil_model;

    const char* mode_env = std::getenv("PUPIL_DETECT_MODE");
    const std::string mode = mode_env && mode_env[0]
        ? std::string(mode_env) : default_mode;
    if (mode == "classic") config.tracking.pupil_mode = PupilMode::Classic;
    else if (mode == "model") config.tracking.pupil_mode = PupilMode::Model;
    else config.tracking.pupil_mode = PupilMode::Hybrid;

    printf("[ALGO] pupil_mode=%s scrfd_tracking_hz=%.1f "
           "scrfd_degraded_hz=%.1f pupil_conf_min=%.2f "
           "pupil_kf=%d pupil_pred=%d gaze_kf=%d gaze_pred=%d "
           "one_euro=(%.2f,%.2f,%.2f)\n",
           mode.c_str(), config.tracking.scrfd_tracking_hz,
           config.tracking.scrfd_degraded_hz,
           config.tracking.pupil_confidence_min,
           config.tracking.pupil_kalman_enabled ? 1 : 0,
           config.tracking.pupil_prediction_hold_frames,
           config.attention.gaze_kalman_enabled ? 1 : 0,
           config.attention.gaze_prediction_hold_frames,
           config.attention.one_euro_min_cutoff,
           config.attention.one_euro_beta,
           config.attention.one_euro_derivative_cutoff);
    return config;
}

void AppendSevenSegmentDigit(std::vector<RectF>& boxes, int digit,
                             float x, float y, float scale) {
    static const uint8_t masks[10] = {
        0x3f, 0x06, 0x5b, 0x4f, 0x66,
        0x6d, 0x7d, 0x07, 0x7f, 0x6f
    };
    static const RectF segments[7] = {
        {{1.f, 0.f, 5.f, 1.f}}, {{5.f, 1.f, 6.f, 5.f}},
        {{5.f, 6.f, 6.f, 10.f}}, {{1.f, 10.f, 5.f, 11.f}},
        {{0.f, 6.f, 1.f, 10.f}}, {{0.f, 1.f, 1.f, 5.f}},
        {{1.f, 5.f, 5.f, 6.f}}
    };
    digit = std::max(0, std::min(9, digit));
    for (int i = 0; i < 7; ++i) {
        if ((masks[digit] & (1u << i)) == 0) continue;
        boxes.push_back(RectF{{
            x + segments[i][0] * scale,
            y + segments[i][1] * scale,
            x + segments[i][2] * scale,
            y + segments[i][3] * scale}});
    }
}

void AppendCalibrationProgress(std::vector<RectF>& boxes,
                               const CalibrationState& calibration) {
    const int point = calibration.point_index + 1;
    const int samples = calibration.sample_count;
    AppendSevenSegmentDigit(boxes, point, 530.f, 15.f, 2.f);
    AppendSevenSegmentDigit(boxes, samples / 10, 560.f, 15.f, 2.f);
    AppendSevenSegmentDigit(boxes, samples % 10, 578.f, 15.f, 2.f);
}

}  // namespace

class EyeTrackingApp::Impl {
public:
    explicit Impl(const char* default_pupil_mode)
        : default_pupil_mode_(default_pupil_mode ? default_pupil_mode : "hybrid") {}

    bool Initialize() {
        if (initialized_) return true;
        if (ssne_initial()) {
            fprintf(stderr, "SSNE init failed\n");
            return false;
        }
        ssne_ready_ = true;
        config_ = LoadTaskConfig(default_pupil_mode_);

        std::array<int, 2> display_shape{{kImageWidth, kImageHeight * 2}};
        visualizer_.Initialize(display_shape);
        visualizer_ready_ = true;

        std::array<int, 2> image_shape{{kImageWidth, kImageHeight}};
        if (!processor_.Initialize(&image_shape)) {
            fprintf(stderr, "[APP] camera pipeline initialization failed, ret=%d\n",
                    processor_.LastError());
            Shutdown();
            return false;
        }
        processor_ready_ = true;

        if (!task_.Initialize(config_)) {
            fprintf(stderr, "EyeTrackingTask init failed\n");
            Shutdown();
            return false;
        }
        task_ready_ = true;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        task_.HandleCommand(TaskCommand::StartCalibration);

        for (int i = 0; i < kFramePoolSize; ++i) {
            frame_pool_[i] = create_tensor(kImageWidth, kImageHeight,
                                           SSNE_Y_8, SSNE_BUF_AI);
            if (!TensorHasCapacity(frame_pool_[i],
                                   static_cast<size_t>(kImageWidth) * kImageHeight)) {
                fprintf(stderr, "[APP] frame pool allocation failed, index=%d\n", i);
                Shutdown();
                return false;
            }
            frame_pool_allocated_++;
        }
        frame_pool_ready_ = true;
        free_frame_mask_.store((1u << kFramePoolSize) - 1u);

        output_sensor_[0] = create_tensor(kImageWidth, kImageHeight * 2,
                                          SSNE_Y_8, SSNE_BUF_AI);
        if (!TensorHasCapacity(output_sensor_[0],
                               static_cast<size_t>(kImageWidth) * kImageHeight * 2)) {
            fprintf(stderr, "[APP] display output allocation failed, index=0\n");
            Shutdown();
            return false;
        }
        display_tensor_mask_ |= 1u << 0;
        output_sensor_[1] = create_tensor(kImageWidth, kImageHeight * 2,
                                          SSNE_Y_8, SSNE_BUF_AI);
        if (!TensorHasCapacity(output_sensor_[1],
                               static_cast<size_t>(kImageWidth) * kImageHeight * 2)) {
            fprintf(stderr, "[APP] display output allocation failed, index=1\n");
            Shutdown();
            return false;
        }
        display_tensor_mask_ |= 1u << 1;
        mirror_display_[0] = create_tensor(kImageWidth, kImageHeight,
                                           SSNE_Y_8, SSNE_BUF_AI);
        if (!TensorHasCapacity(mirror_display_[0],
                               static_cast<size_t>(kImageWidth) * kImageHeight)) {
            fprintf(stderr, "[APP] mirror tensor allocation failed, index=0\n");
            Shutdown();
            return false;
        }
        display_tensor_mask_ |= 1u << 2;
        mirror_display_[1] = create_tensor(kImageWidth, kImageHeight,
                                           SSNE_Y_8, SSNE_BUF_AI);
        if (!TensorHasCapacity(mirror_display_[1],
                               static_cast<size_t>(kImageWidth) * kImageHeight)) {
            fprintf(stderr, "[APP] mirror tensor allocation failed, index=1\n");
            Shutdown();
            return false;
        }
        display_tensor_mask_ |= 1u << 3;
        display_tensors_ready_ = true;
        initialized_ = true;
        return true;
    }

    int Run() {
        if (!initialized_) return -1;
        ssne_tensor_t image_sensor[2];
        bool initial_frame_ready = false;
        for (int attempt = 0; attempt < kInitialCaptureAttempts; ++attempt) {
            if (CaptureWithRecovery(&image_sensor[0], &image_sensor[1])) {
                initial_frame_ready = true;
                break;
            }
        }
        if (!initial_frame_ready) {
            fprintf(stderr,
                    "[CAMERA] unable to acquire initial frame after %d attempts\n",
                    kInitialCaptureAttempts);
            return -2;
        }
        copy_double_tensor_buffer(image_sensor[0], image_sensor[1], output_sensor_[0]);
        copy_double_tensor_buffer(image_sensor[0], image_sensor[1], output_sensor_[1]);
        set_isp_debug_config(output_sensor_[0], output_sensor_[1]);
        start_isp_debug_load();
        start_isp_debug_load();

        stopping_.store(false);
        exit_requested_.store(false);
        try {
            inference_thread_ = std::thread(&Impl::InferenceLoop, this);
            input_thread_ = std::thread(&Impl::InputLoop, this);
        } catch (const std::exception& error) {
            fprintf(stderr, "[APP] worker thread startup failed: %s\n",
                    error.what());
            stopping_.store(true);
            exit_requested_.store(true);
            queue_cv_.notify_all();
            if (inference_thread_.joinable()) inference_thread_.join();
            return -3;
        }

        uint8_t load_flag = 0;
        uint64_t frame_id = 0;
        while (!exit_requested_.load()) {
            if (!CaptureWithRecovery(&image_sensor[0], &image_sensor[1])) {
                LogPerformance();
                continue;
            }
            const TimePoint captured_at = Clock::now();
            captured_.fetch_add(1);
            get_even_or_odd_flag(load_flag);
            mirror_tensor(image_sensor[0], mirror_display_[0]);
            mirror_tensor(image_sensor[1], mirror_display_[1]);
            if (load_flag == 0)
                copy_double_tensor_buffer(mirror_display_[0], mirror_display_[1],
                                          output_sensor_[0]);
            else
                copy_double_tensor_buffer(mirror_display_[0], mirror_display_[1],
                                          output_sensor_[1]);
            start_isp_debug_load();

            if (TryEnqueue(image_sensor[0], frame_id, captured_at))
                enqueued_.fetch_add(1);
            else
                dropped_.fetch_add(1);
            frame_id++;
            LogPerformance();
        }

        if (input_thread_.joinable()) input_thread_.join();
        stopping_.store(true);
        queue_cv_.notify_one();
        if (inference_thread_.joinable()) inference_thread_.join();
        return 0;
    }

    void Shutdown() {
        if (!ssne_ready_ && !initialized_) return;
        exit_requested_.store(true);
        stopping_.store(true);
        queue_cv_.notify_all();
        if (inference_thread_.joinable()) inference_thread_.join();
        if (input_thread_.joinable()) input_thread_.join();

        if (task_ready_) {
            task_.Release();
            task_ready_ = false;
        }
        if (display_tensor_mask_ & (1u << 2)) release_tensor(mirror_display_[0]);
        if (display_tensor_mask_ & (1u << 3)) release_tensor(mirror_display_[1]);
        if (display_tensor_mask_ & (1u << 0)) release_tensor(output_sensor_[0]);
        if (display_tensor_mask_ & (1u << 1)) release_tensor(output_sensor_[1]);
        display_tensor_mask_ = 0;
        display_tensors_ready_ = false;
        for (int i = 0; i < frame_pool_allocated_; ++i)
            release_tensor(frame_pool_[i]);
        frame_pool_allocated_ = 0;
        frame_pool_ready_ = false;
        if (processor_ready_) {
            processor_.Release();
            processor_ready_ = false;
        }
        if (visualizer_ready_) {
            visualizer_.Release();
            visualizer_ready_ = false;
        }
        if (ssne_ready_) {
            if (ssne_release()) fprintf(stderr, "SSNE release failed\n");
            ssne_ready_ = false;
        }
        initialized_ = false;
    }

private:
    static const int kImageWidth = 640;
    static const int kImageHeight = 480;
    static const int kFramePoolSize = 2;
    static const int kMaxBlinkMarks = 20;
    static const int kInitialCaptureAttempts = 10;
    static const int kCaptureFailureRestartThreshold = 3;

    struct InferenceFrame {
        int pool_index = -1;
        uint64_t frame_id = 0;
        TimePoint captured_at;
    };

    bool CaptureWithRecovery(ssne_tensor_t* left, ssne_tensor_t* right) {
        if (processor_.GetDualImage(left, right)) {
            consecutive_capture_failures_ = 0;
            return true;
        }

        capture_failures_.fetch_add(1);
        consecutive_capture_failures_++;
        const int error_code = processor_.LastError();
        fprintf(stderr, "[CAMERA] capture failed, ret=%d streak=%d\n",
                error_code, consecutive_capture_failures_);

        if (consecutive_capture_failures_ >=
            kCaptureFailureRestartThreshold) {
            pipeline_restart_attempts_.fetch_add(1);
            if (processor_.Restart()) {
                pipeline_restart_successes_.fetch_add(1);
                consecutive_capture_failures_ = 0;
                reset_requested_.store(true);
                fprintf(stderr, "[CAMERA] pipeline restart succeeded\n");
            } else {
                pipeline_restart_failures_.fetch_add(1);
                fprintf(stderr, "[CAMERA] pipeline restart failed, ret=%d\n",
                        processor_.LastError());
            }
        }

        const int shift = std::min(6, std::max(0,
            consecutive_capture_failures_ - 1));
        const int backoff_ms = std::min(200, 2 * (1 << shift));
        std::this_thread::sleep_for(std::chrono::milliseconds(backoff_ms));
        return false;
    }

    int AcquireFrameSlot() {
        unsigned int mask = free_frame_mask_.load(std::memory_order_relaxed);
        while (mask != 0) {
            int slot = 0;
            while ((mask & (1u << slot)) == 0u) slot++;
            const unsigned int updated = mask & ~(1u << slot);
            if (free_frame_mask_.compare_exchange_weak(
                    mask, updated, std::memory_order_acquire,
                    std::memory_order_relaxed)) return slot;
        }
        return -1;
    }

    void ReturnFrameSlot(int slot) {
        if (slot >= 0) free_frame_mask_.fetch_or(1u << slot,
                                                 std::memory_order_release);
    }

    bool TryEnqueue(ssne_tensor_t& source, uint64_t frame_id,
                    TimePoint captured_at) {
        int slot = AcquireFrameSlot();
        if (slot < 0) {
            std::unique_lock<std::mutex> recycle_lock(queue_mutex_,
                                                       std::try_to_lock);
            if (!recycle_lock.owns_lock() || frame_queue_.empty()) {
                queue_busy_drops_.fetch_add(1);
                return false;
            }
            slot = frame_queue_.front().pool_index;
            frame_queue_.pop();
            queue_depth_.store(0);
            dropped_.fetch_add(1);
            stale_replacements_.fetch_add(1);
        }
        if (copy_tensor_buffer(source, frame_pool_[slot]) !=
            SSNE_ERRCODE_NO_ERROR) {
            frame_copy_failures_.fetch_add(1);
            ReturnFrameSlot(slot);
            return false;
        }

        std::unique_lock<std::mutex> lock(queue_mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            queue_busy_drops_.fetch_add(1);
            ReturnFrameSlot(slot);
            return false;
        }
        if (!frame_queue_.empty()) {
            const InferenceFrame stale = frame_queue_.front();
            frame_queue_.pop();
            ReturnFrameSlot(stale.pool_index);
            dropped_.fetch_add(1);
            stale_replacements_.fetch_add(1);
        }
        InferenceFrame frame;
        frame.pool_index = slot;
        frame.frame_id = frame_id;
        frame.captured_at = captured_at;
        frame_queue_.push(frame);
        queue_depth_.store(1);
        lock.unlock();
        queue_cv_.notify_one();
        return true;
    }

    void InferenceLoop() {
        printf("[Thread] Inference started\n");
        while (true) {
            InferenceFrame frame;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return !frame_queue_.empty() || stopping_.load();
                });
                if (stopping_.load() && frame_queue_.empty()) break;
                frame = frame_queue_.front();
                frame_queue_.pop();
                queue_depth_.store(0);
            }

            bool processed = false;
            try {
                if (calibration_requested_.exchange(false))
                    task_.HandleCommand(TaskCommand::StartCalibration);
                if (reset_requested_.exchange(false))
                    task_.HandleCommand(TaskCommand::Reset);
                if (clear_marks_requested_.exchange(false)) {
                    blink_marks_.clear();
                    std::vector<RectF> empty;
                    visualizer_.Draw(empty);
                }

                FramePacket packet;
                packet.image = &frame_pool_[frame.pool_index];
                packet.frame_id = frame.frame_id;
                packet.captured_at = frame.captured_at;
                const EyeTrackingResult result = task_.Process(packet);
                if (result.tracking.face.detector_scheduled)
                    detector_requests_.fetch_add(1);
                if (result.tracking.face.detector_ran)
                    detector_runs_.fetch_add(1);
                if (result.tracking.face.detector_ran) {
                    const uint64_t detector_latency = static_cast<uint64_t>(
                        std::max(0.f,
                            result.tracking.face.detector_latency_ms) * 1000.f);
                    detector_latency_us_.fetch_add(detector_latency);
                    UpdateMaxDetectorLatency(detector_latency);
                }
                if (result.tracking.left_pupil.used_model)
                    model_runs_.fetch_add(1);
                if (result.tracking.right_pupil.used_model)
                    model_runs_.fetch_add(1);
                if (result.tracking.frame_data_error)
                    frame_data_failures_.fetch_add(1);
                if (result.tracking.face.detector_error)
                    detector_failures_.fetch_add(1);
                if (result.tracking.left_pupil.model_error)
                    model_failures_.fetch_add(1);
                if (result.tracking.right_pupil.model_error)
                    model_failures_.fetch_add(1);
                if (result.attention.gaze_valid) gaze_valid_.fetch_add(1);
                if (result.tracking.blink.event && result.attention.gaze_valid) {
                    if (static_cast<int>(blink_marks_.size()) >= kMaxBlinkMarks)
                        blink_marks_.erase(blink_marks_.begin());
                    blink_marks_.push_back(result.attention.gaze);
                }

                {
                    std::lock_guard<std::mutex> lock(result_mutex_);
                    latest_result_ = result;
                }
                Render(result);
                processed = true;
            } catch (const std::exception& error) {
                processing_failures_.fetch_add(1);
                dropped_.fetch_add(1);
                fprintf(stderr, "[Inference] processing exception: %s\n",
                        error.what());
            } catch (...) {
                processing_failures_.fetch_add(1);
                dropped_.fetch_add(1);
                fprintf(stderr, "[Inference] unknown processing exception\n");
            }

            if (processed) {
                const uint64_t latency = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        Clock::now() - frame.captured_at).count());
                latency_us_.fetch_add(latency);
                UpdateMaxLatency(latency);
                inferred_.fetch_add(1);
            }
            ReturnFrameSlot(frame.pool_index);
        }
        printf("[Thread] Inference stopped\n");
    }

    void InputLoop() {
        std::string input;
        printf("Commands: q=quit c=calibrate n=status r=clear marks x=reset\n");
        while (!exit_requested_.load()) {
#ifdef __linux__
            fd_set read_set;
            FD_ZERO(&read_set);
            FD_SET(STDIN_FILENO, &read_set);
            timeval timeout;
            timeout.tv_sec = 0;
            timeout.tv_usec = 200000;
            const int ready = select(STDIN_FILENO + 1, &read_set,
                                     nullptr, nullptr, &timeout);
            if (ready < 0) {
                if (errno == EINTR) continue;
                fprintf(stderr, "[Input] select failed, errno=%d\n", errno);
                break;
            }
            if (ready == 0) continue;
#endif
            if (!(std::cin >> input)) break;
            if (input == "q" || input == "Q") {
                exit_requested_.store(true);
                break;
            }
            if (input == "c" || input == "C") {
                calibration_requested_.store(true);
            } else if (input == "r" || input == "R") {
                clear_marks_requested_.store(true);
            } else if (input == "x" || input == "X") {
                reset_requested_.store(true);
            } else if (input == "n" || input == "N") {
                std::lock_guard<std::mutex> lock(result_mutex_);
                const CalibrationState& calibration =
                    latest_result_.attention.calibration;
                printf("[Status] mode=%s face=%d gaze=%d calibration=%d "
                       "point=%d/9 samples=%d\n",
                       TrackingModeName(latest_result_.tracking.face.mode),
                       latest_result_.tracking.face.valid ? 1 : 0,
                       latest_result_.attention.gaze_valid ? 1 : 0,
                       calibration.active ? 1 : 0,
                       calibration.point_index + 1,
                       calibration.sample_count);
            }
        }
        exit_requested_.store(true);
    }

    void Render(const EyeTrackingResult& result) {
        std::vector<RectF> boxes;
        boxes.reserve(32);
        if (result.tracking.face.valid)
            boxes.push_back(result.tracking.face.box);
        if (result.tracking.left_pupil.valid) {
            const float x = result.tracking.left_pupil.position[0];
            const float y = result.tracking.left_pupil.position[1];
            boxes.push_back(RectF{{x - 8.f, y - 8.f, x + 8.f, y + 8.f}});
        }
        if (result.tracking.right_pupil.valid) {
            const float x = result.tracking.right_pupil.position[0];
            const float y = result.tracking.right_pupil.position[1];
            boxes.push_back(RectF{{x - 8.f, y - 8.f, x + 8.f, y + 8.f}});
        }
        boxes.push_back(RectF{{0.f, 480.f, 640.f, 960.f}});

        if (result.attention.gaze_valid) {
            const float x = result.attention.gaze[0] * kImageWidth;
            const float y = 480.f + result.attention.gaze[1] * kImageHeight;
            boxes.push_back(RectF{{x - 12.f, y - 12.f, x + 12.f, y + 12.f}});

            int column = 1;
            int row = 1;
            switch (result.attention.direction) {
                case GazeDirection::Left:
                case GazeDirection::LeftUp:
                case GazeDirection::LeftDown: column = 0; break;
                case GazeDirection::Right:
                case GazeDirection::RightUp:
                case GazeDirection::RightDown: column = 2; break;
                default: break;
            }
            switch (result.attention.direction) {
                case GazeDirection::Up:
                case GazeDirection::LeftUp:
                case GazeDirection::RightUp: row = 0; break;
                case GazeDirection::Down:
                case GazeDirection::LeftDown:
                case GazeDirection::RightDown: row = 2; break;
                default: break;
            }
            const float cell = 18.f;
            const float x1 = 10.f + column * cell;
            const float y1 = 10.f + row * cell;
            boxes.push_back(RectF{{x1, y1, x1 + cell - 2.f, y1 + cell - 2.f}});
        }

        for (size_t i = 0; i < blink_marks_.size(); ++i) {
            const float x = blink_marks_[i][0] * kImageWidth;
            const float y = blink_marks_[i][1] * kImageHeight + 480.f;
            boxes.push_back(RectF{{x - 5.f, y - 5.f, x + 5.f, y + 5.f}});
        }
        if (result.attention.calibration.active) {
            const float x = result.attention.calibration.target[0] * kImageWidth;
            const float y = result.attention.calibration.target[1] * kImageHeight;
            boxes.push_back(RectF{{x - 14.f, y - 14.f, x + 14.f, y + 14.f}});
            AppendCalibrationProgress(boxes, result.attention.calibration);
        }
        visualizer_.Draw(boxes);
    }

    void UpdateMaxLatency(uint64_t latency) {
        uint64_t current = latency_max_us_.load();
        while (current < latency &&
               !latency_max_us_.compare_exchange_weak(current, latency)) {}
    }

    void UpdateMaxDetectorLatency(uint64_t latency) {
        uint64_t current = detector_latency_max_us_.load();
        while (current < latency &&
               !detector_latency_max_us_.compare_exchange_weak(
                   current, latency)) {}
    }

    void LogPerformance() {
        const TimePoint now = Clock::now();
        const double elapsed = std::chrono::duration<double>(now - last_log_).count();
        if (elapsed < 1.0) return;
        const uint64_t captured = captured_.load();
        const uint64_t enqueued = enqueued_.load();
        const uint64_t dropped = dropped_.load();
        const uint64_t inferred = inferred_.load();
        const uint64_t latency = latency_us_.load();
        const uint64_t detector = detector_runs_.load();
        const uint64_t detector_requests = detector_requests_.load();
        const uint64_t detector_latency = detector_latency_us_.load();
        const uint64_t model = model_runs_.load();
        const uint64_t gaze = gaze_valid_.load();
        const uint64_t inferred_delta = inferred - last_inferred_;
        const uint64_t latency_delta = latency - last_latency_;
        const double average_latency = inferred_delta == 0 ? 0.0
            : static_cast<double>(latency_delta) / inferred_delta / 1000.0;
        const uint64_t max_latency = latency_max_us_.exchange(0);
        const uint64_t detector_delta = detector - last_detector_;
        const uint64_t detector_latency_delta =
            detector_latency - last_detector_latency_;
        const double average_detector_latency = detector_delta == 0 ? 0.0
            : static_cast<double>(detector_latency_delta) /
              detector_delta / 1000.0;
        const uint64_t max_detector_latency =
            detector_latency_max_us_.exchange(0);

        TrackingMode mode = TrackingMode::Reacquire;
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            mode = latest_result_.tracking.face.mode;
        }
        printf("[PERF] capture_fps=%.1f enqueue_fps=%.1f drop=%llu "
               "epp_fps=%.1f gaze=%llu scrfd_req=%llu scrfd_done=%llu "
               "scrfd_ms_avg=%.1f scrfd_ms_max=%.1f model_recovery=%llu "
               "queue=%d latency_ms_avg=%.1f latency_ms_max=%.1f mode=%s\n",
               (captured - last_captured_) / elapsed,
               (enqueued - last_enqueued_) / elapsed,
               static_cast<unsigned long long>(dropped - last_dropped_),
               inferred_delta / elapsed,
               static_cast<unsigned long long>(gaze - last_gaze_),
               static_cast<unsigned long long>(
                   detector_requests - last_detector_requests_),
               static_cast<unsigned long long>(detector_delta),
               average_detector_latency, max_detector_latency / 1000.0,
               static_cast<unsigned long long>(model - last_model_),
               queue_depth_.load(), average_latency, max_latency / 1000.0,
               TrackingModeName(mode));
        printf("[HEALTH] capture_fail=%llu restart=%llu/%llu restart_fail=%llu "
               "copy_fail=%llu queue_busy=%llu stale=%llu frame_data=%llu "
               "scrfd_fail=%llu pupil_model_fail=%llu process_fail=%llu\n",
               static_cast<unsigned long long>(capture_failures_.load()),
               static_cast<unsigned long long>(pipeline_restart_successes_.load()),
               static_cast<unsigned long long>(pipeline_restart_attempts_.load()),
               static_cast<unsigned long long>(pipeline_restart_failures_.load()),
               static_cast<unsigned long long>(frame_copy_failures_.load()),
               static_cast<unsigned long long>(queue_busy_drops_.load()),
               static_cast<unsigned long long>(stale_replacements_.load()),
               static_cast<unsigned long long>(frame_data_failures_.load()),
               static_cast<unsigned long long>(detector_failures_.load()),
               static_cast<unsigned long long>(model_failures_.load()),
               static_cast<unsigned long long>(processing_failures_.load()));

        last_captured_ = captured;
        last_enqueued_ = enqueued;
        last_dropped_ = dropped;
        last_inferred_ = inferred;
        last_latency_ = latency;
        last_detector_ = detector;
        last_detector_requests_ = detector_requests;
        last_detector_latency_ = detector_latency;
        last_model_ = model;
        last_gaze_ = gaze;
        last_log_ = now;
    }

    std::string default_pupil_mode_;
    TaskConfig config_;
    EyeTrackingTask task_;
    IMAGEPROCESSOR processor_;
    VISUALIZER visualizer_;

    std::array<ssne_tensor_t, kFramePoolSize> frame_pool_;
    std::array<ssne_tensor_t, 2> output_sensor_;
    std::array<ssne_tensor_t, 2> mirror_display_;
    std::atomic<unsigned int> free_frame_mask_{0};
    std::queue<InferenceFrame> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<int> queue_depth_{0};

    std::thread inference_thread_;
    std::thread input_thread_;
    std::atomic<bool> stopping_{false};
    std::atomic<bool> exit_requested_{false};
    std::atomic<bool> calibration_requested_{false};
    std::atomic<bool> clear_marks_requested_{false};
    std::atomic<bool> reset_requested_{false};

    EyeTrackingResult latest_result_;
    std::mutex result_mutex_;
    std::vector<Point2f> blink_marks_;

    std::atomic<uint64_t> captured_{0};
    std::atomic<uint64_t> enqueued_{0};
    std::atomic<uint64_t> dropped_{0};
    std::atomic<uint64_t> inferred_{0};
    std::atomic<uint64_t> latency_us_{0};
    std::atomic<uint64_t> latency_max_us_{0};
    std::atomic<uint64_t> detector_runs_{0};
    std::atomic<uint64_t> detector_requests_{0};
    std::atomic<uint64_t> detector_latency_us_{0};
    std::atomic<uint64_t> detector_latency_max_us_{0};
    std::atomic<uint64_t> model_runs_{0};
    std::atomic<uint64_t> gaze_valid_{0};
    std::atomic<uint64_t> capture_failures_{0};
    std::atomic<uint64_t> pipeline_restart_attempts_{0};
    std::atomic<uint64_t> pipeline_restart_successes_{0};
    std::atomic<uint64_t> pipeline_restart_failures_{0};
    std::atomic<uint64_t> frame_copy_failures_{0};
    std::atomic<uint64_t> queue_busy_drops_{0};
    std::atomic<uint64_t> stale_replacements_{0};
    std::atomic<uint64_t> frame_data_failures_{0};
    std::atomic<uint64_t> detector_failures_{0};
    std::atomic<uint64_t> model_failures_{0};
    std::atomic<uint64_t> processing_failures_{0};

    TimePoint last_log_ = Clock::now();
    uint64_t last_captured_ = 0;
    uint64_t last_enqueued_ = 0;
    uint64_t last_dropped_ = 0;
    uint64_t last_inferred_ = 0;
    uint64_t last_latency_ = 0;
    uint64_t last_detector_ = 0;
    uint64_t last_detector_requests_ = 0;
    uint64_t last_detector_latency_ = 0;
    uint64_t last_model_ = 0;
    uint64_t last_gaze_ = 0;

    bool initialized_ = false;
    bool ssne_ready_ = false;
    bool visualizer_ready_ = false;
    bool processor_ready_ = false;
    bool task_ready_ = false;
    bool frame_pool_ready_ = false;
    bool display_tensors_ready_ = false;
    int frame_pool_allocated_ = 0;
    unsigned int display_tensor_mask_ = 0;
    int consecutive_capture_failures_ = 0;
};

EyeTrackingApp::EyeTrackingApp(const char* default_pupil_mode)
    : impl_(new Impl(default_pupil_mode)) {}

EyeTrackingApp::~EyeTrackingApp() { impl_->Shutdown(); }

bool EyeTrackingApp::Initialize() { return impl_->Initialize(); }
int EyeTrackingApp::Run() { return impl_->Run(); }
void EyeTrackingApp::Shutdown() { impl_->Shutdown(); }

}  // namespace eye_track
