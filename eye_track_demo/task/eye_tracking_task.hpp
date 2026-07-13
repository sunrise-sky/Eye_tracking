#pragma once

#include "attention_task.hpp"
#include "tracking_task.hpp"

namespace eye_track {

class EyeTrackingTask {
public:
    bool Initialize(const TaskConfig& config);
    EyeTrackingResult Process(const FramePacket& frame);
    void HandleCommand(TaskCommand command);
    void Release();

private:
    TrackingTask tracking_task_;
    AttentionTask attention_task_;
    bool initialized_ = false;
};

}  // namespace eye_track
