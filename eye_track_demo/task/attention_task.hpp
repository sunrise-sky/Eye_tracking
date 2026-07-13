#pragma once

#include <memory>

#include "task_types.hpp"

namespace eye_track {

class AttentionTask {
public:
    AttentionTask();
    ~AttentionTask();

    bool Initialize(const AttentionConfig& config);
    AttentionState Process(const TrackingState& tracking);
    void StartCalibration();
    void Reset();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eye_track
