#include "eye_tracking_app.hpp"

#include <exception>
#include <cstdio>

int main() {
#ifdef EYE_TRACK_DEFAULT_CLASSIC
    const char* default_mode = "classic";
#else
    const char* default_mode = "hybrid";
#endif

    try {
        eye_track::EyeTrackingApp app(default_mode);
        if (!app.Initialize()) return -1;
        const int result = app.Run();
        app.Shutdown();
        return result;
    } catch (const std::exception& error) {
        fprintf(stderr, "[APP] fatal exception: %s\n", error.what());
    } catch (...) {
        fprintf(stderr, "[APP] unknown fatal exception\n");
    }
    return -4;
}
