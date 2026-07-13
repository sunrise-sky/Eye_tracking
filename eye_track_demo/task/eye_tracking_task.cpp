#include "eye_tracking_task.hpp"

namespace eye_track {

bool EyeTrackingTask::Initialize(const TaskConfig& config) {
    if (!tracking_task_.Initialize(config.tracking)) return false;
    if (!attention_task_.Initialize(config.attention)) {
        tracking_task_.Release();
        return false;
    }
    initialized_ = true;
    return true;
}

EyeTrackingResult EyeTrackingTask::Process(const FramePacket& frame) {
    EyeTrackingResult result;
    if (!initialized_) return result;
    result.tracking = tracking_task_.Process(frame);
    result.attention = attention_task_.Process(result.tracking);
    return result;
}

void EyeTrackingTask::HandleCommand(TaskCommand command) {
    switch (command) {
        case TaskCommand::StartCalibration:
            attention_task_.StartCalibration();
            break;
        case TaskCommand::Reset:
            tracking_task_.Reset();
            attention_task_.Reset();
            break;
        case TaskCommand::ClearMarks:
            break;
    }
}

void EyeTrackingTask::Release() {
    if (!initialized_) return;
    tracking_task_.Release();
    initialized_ = false;
}

}  // namespace eye_track
