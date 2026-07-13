#include "eye_tracking_app.hpp"

int main() {
#ifdef EYE_TRACK_DEFAULT_CLASSIC
    const char* default_mode = "classic";
#else
    const char* default_mode = "hybrid";
#endif

    eye_track::EyeTrackingApp app(default_mode);
    if (!app.Initialize()) return -1;
    const int result = app.Run();
    app.Shutdown();
    return result;
}
