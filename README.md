# Eye Tracking

## Overview

Eye Tracking is an embedded vision project for real-time gaze interaction. It captures camera frames, locates the face and eye regions, estimates pupil movement, maps the result into a normalized gaze position, and renders lightweight visual feedback on screen.

The project is intended as a compact demonstration of how an eye-tracking pipeline can be organized on an embedded device. It focuses on the overall interaction flow rather than exposing hardware-specific, model-specific, or personal project details.

## High-Level Framework

The system is organized around a simple real-time pipeline:

```text
Camera input
  -> image acquisition
  -> face and eye region detection
  -> pupil position estimation
  -> gaze calibration and smoothing
  -> blink/event detection
  -> on-screen visual feedback
```

The runtime application separates frame acquisition from inference work. The main loop keeps image capture and display responsive, while a background processing thread handles vision analysis and gaze estimation. A small frame queue connects the two parts so the application can keep running smoothly even when processing takes longer than one frame interval.

## Main Components

- `eye_track_demo/`: embedded runtime demo, including image capture, processing logic, gaze estimation, and display overlay code.
- `training/`: offline training and model preparation materials used during development.

## Design Ideas

- Keep the display loop responsive by avoiding long blocking work in the main capture path.
- Use face localization to restrict eye and pupil analysis to a smaller region of interest.
- Convert pupil offsets into a normalized gaze coordinate through calibration.
- Smooth recent gaze estimates to reduce jitter.
- Treat blink detection as an interaction event that can mark the current gaze position.

This repository is structured as a project prototype and reference implementation for an embedded gaze-interaction workflow.
