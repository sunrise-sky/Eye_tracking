#!/bin/sh

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$APP_DIR" || exit 1

if [ -n "${EYE_TRACK_APP:-}" ]; then
    APP="$EYE_TRACK_APP"
elif [ -f ./ssne_ai_demo_model ]; then
    APP=./ssne_ai_demo_model
else
    APP=./ssne_ai_demo
fi

if [ ! -f "$APP" ]; then
    echo "eye tracking executable not found: $APP" >&2
    exit 1
fi

chmod +x "$APP"
exec "$APP"
