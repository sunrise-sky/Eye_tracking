#pragma once

#include <memory>

#include "task_types.hpp"

namespace eye_track {

class TrackingTask {
public:
    TrackingTask();
    ~TrackingTask();

    bool Initialize(const TrackingConfig& config);
    TrackingState Process(const FramePacket& frame);
    void Reset();
    void Release();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eye_track
