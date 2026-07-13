#!/bin/sh

APP="${EYE_TRACK_APP:-./ssne_ai_demo_model}"
chmod +x "$APP"
exec "$APP"
