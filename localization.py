from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator
import pyrealsense2 as rs

class_colors = {
    0: (255, 0, 0),
    1: (0, 140, 255),
    2: (0, 165, 255),
    3: (128, 128, 128),
    4: (0, 255, 255)
}

def initialize_pipeline():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics
    align = rs.align(rs.stream.color)

    return pipeline, align, intrinsics, depth_scale

def get_frames(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return depth_image, color_image

def compute_bboxes_masks(color_image, boxes, masks, model_names):
    annotator = Annotator(color_image)
    centroids = []
    for box, mask in zip(boxes, masks):
        b = box.xyxy[0]
        c = box.cls
        color = class_colors[int(c)]

        annotator.box_label(b, model_names[int(c)])

        if masks is not None and len(masks.xy) > 0:
            contour = mask.xy[0].astype(np.int32).reshape(-1, 1, 2)
            overlay = np.zeros_like(color_image, dtype=np.uint8)
            cv2.fillPoly(overlay, [contour], color)
            cv2.addWeighted(overlay, 0.6, color_image, 1.0, 0, color_image)
            cv2.drawContours(color_image, [contour], -1, color, 2)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(color_image, (cx, cy), 5, (255, 255, 255), -1)
                centroids.append((cx, cy))
    return annotator.result(), centroids

def extract_depth(depth_frame, cx, cy):
    if 0 <= cx < depth_frame.shape[1] and 0 <= cy < depth_frame.shape[0]:
        depth_value = depth_frame[cy, cx]
        return depth_value
    return None

def depth_to_pointcloud(intrinsics, depth_frame, depth_scale):
    depth_frame = depth_frame * depth_scale
    
    rows, cols = depth_frame.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

    r = r.astype(float)
    c = c.astype(float)

    z = depth_frame
    x = z * (c - intrinsics.ppx) / intrinsics.fx
    y = z * (r - intrinsics.ppy) / intrinsics.fy

    z = np.ravel(z)
    x = np.ravel(x)
    y = np.ravel(y)

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    return points


def main():
    pipeline, align, intrinsics, depth_scale = initialize_pipeline()
    model = YOLO('./best_seg.onnx', task="segment")

    try:
        while True:
            depth_image, color_image = get_frames(pipeline, align)
            pointcloud = depth_to_pointcloud(intrinsics, depth_image, depth_scale)
            if color_image is None or depth_image is None:
                continue

            res = model.predict(color_image, imgsz=(480, 640))
            model_names = model.names
            for r in res:
                if r.boxes.shape[0] != 0:
                    color_image, centroids = compute_bboxes_masks(color_image, r.boxes, r.masks, model_names)
                    for cx, cy in centroids:
                        depth = extract_depth(depth_image, cx, cy)
                        if depth is not None:
                            cv2.circle(depth_image, (cx, cy), 10, (255, 255, 255), -1)
                            point = pointcloud[cy * depth_image.shape[1] + cx]
                            print(f"Point: {point}")
                            print(f"Depth: {depth}")
                else:
                    continue

            depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_image))

            cv2.imshow('YOLO V8 Segmentation', images)

            if cv2.waitKey(1) & 0xFF == ord(' '):
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

if __name__ == "__main__":
    main()
