# Keyframe-Based RGB-D SLAM with Local Bundle Adjustment

This repository contains a **lightweight, keyframe-based SLAM layer built on top of a frame-to-frame Visual Odometry (VO) baseline**. It was developed during the *Signal, Images and Videos* course at the University of Trento for the E-Agle Trento Racing Team’s driverless car project. The system provides accurate, real-time vehicle localization using only data from an RGB-D camera.

## General Overview
This SLAM system constructs a map by leveraging two primary entities: *KeyFrames*, which are selected camera frames preserved over time due to their rich and distinctive visual features, and *MapPoints* (also called *landmarks*), which represent the 3D projections of the 2D visual features extracted from the keyframes.

At each iteration, the system acquires the RGB-D frame, and the corresponding mask of static elements of the current frame via a ZMQ subscriber running in a dedicated thread. The frame is then pre-processed by converting it to grayscale, after which visual features are detected and described using the ORB (Oriented FAST and Rotated BRIEF) algorithm. To establish feature correspondences, k-nearest neighbor brute-force matching is performed between the current frame and the most recent keyframe in the map, yielding a set of candidate feature matches. Finally, the relative pose of the camera is estimated using the Perspective-n-Point (PnP) RANSAC algorithm. By chaining successive roto-translations up to the last keyframe, the system is able to compute an estimate of the global camera pose for the current camera frame, which will be inserted in the map as a keyframe based on policy checks.

Since visual odometry (VO) systems inherently suffer from error accumulation and drift, a local Bundle Adjustment (BA) over a sliding window of recent keyframes is executed upon each keyframe insertion in a background thread. After convergence, the optimized poses and landmarks are written back to the map, while outliers are pruned.

## Project Dependencies

This project relies on the following external libraries and tools:

- **[Sophus](https://github.com/strasdat/Sophus)** – used for SE(3) abstractions, instead of relying on `cv::Mat`, to make geometric operations more expressive and efficient
- **[Ceres Solver](https://github.com/ceres-solver/ceres-solver.git)** – used for the non-linear optimization tasks that appear in the backend of the SLAM
- **[OpenCV](https://opencv.org/)** – used to implement the basic visual odometry pipeline (image processing, feature extraction, and tracking)
- **[ZeroMQ (ZMQ)](https://zeromq.org/)** – used to receive data from a dedicated camera-manager wrapper at each iteration of the main loop
- **as-serializers** – dedicated E-Agle github repo to generate protobuf definitions for the structured data received via ZMQ
- **[Rerun](https://rerun.io/)** – used for visualization of the SLAM process and results


## Useful Links
A more specific description of the project can be found [here](https://drive.google.com/file/d/1lOP8qDL-C51Cc4rqHJWv5zp6SnQotsVQ/view?usp=drive_link), while a demonstration of the system's performance is available at this [link](https://tinyurl.com/video-demo-keyframe-based-SLAM). 

