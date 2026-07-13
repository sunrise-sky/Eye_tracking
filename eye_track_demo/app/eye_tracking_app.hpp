#pragma once

#include <memory>

namespace eye_track {

class EyeTrackingApp {
public:
    explicit EyeTrackingApp(const char* default_pupil_mode);
    ~EyeTrackingApp();

    bool Initialize();
    int Run();
    void Shutdown();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace eye_track
