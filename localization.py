from ultralytics import YOLO
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator
import pyrealsense2 as rs
import zmq
import msgpack
import struct

class_colors = {
    0: (255, 0, 0),
    1: (0, 140, 255),
    2: (0, 165, 255),
    3: (128, 128, 128),
    4: (0, 255, 255)
}

yaw = np.deg2rad(0)
cos_yaw = np.cos(yaw)
sin_yaw = np.sin(yaw)
x_translation = -275 + 6.25
y_translation = 0
z_translation = 747
trasformation_matrix = np.array([
    [cos_yaw, 0, sin_yaw, x_translation],
    [0, 1, 0, y_translation],
    [-sin_yaw, 0, cos_yaw, z_translation],
    [0,0,0,1]
])

def init():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics
    align = rs.align(rs.stream.color)

    socket = zmq.Context().socket(zmq.PUB)
    socket.bind("tcp://*:5555")

    return pipeline, align, intrinsics, depth_scale, socket

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
    classes = []
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
                classes.append(int(c))
    return annotator.result(), centroids, classes

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

def cam_to_com(point):
    homogeneous_point = np.append(point,1)
    return np.matmul(trasformation_matrix,homogeneous_point)[:3]


def main():
    pipeline, align, intrinsics, depth_scale, socket = init()
    model = YOLO('./best_seg.onnx', task="segment")

    try:
        while True:
            depth_image, color_image = get_frames(pipeline, align)
            pointcloud = depth_to_pointcloud(intrinsics, depth_image, depth_scale)
            if color_image is None or depth_image is None:
                continue

            res = model.predict(color_image, imgsz=(480, 640))
            model_names = model.names
            points = []
            cls = []
            for r in res:
                if r.boxes.shape[0] != 0:
                    color_image, centroids, classes = compute_bboxes_masks(color_image, r.boxes, r.masks, model_names)
                    for i in range(0, len(centroids) - 1):
                        cx = centroids[i][0]
                        cy = centroids[i][1]
                        depth = extract_depth(depth_image, cx, cy)
                        if depth is not None and depth != 0:
                            cv2.circle(depth_image, (cx, cy), 10, (255, 255, 255), -1)
                            com_point = cam_to_com(pointcloud[cy * depth_image.shape[1] + cx])

                            print(f"Point: {com_point}")
                            print(f"Depth: {depth}")
                            points.append(com_point.tolist())
                            cls.append(classes[i])
                            print(f"Class ID: {classes[i]}")
                            
                else:
                    continue

            # msg = struct.pack(f"{len(points) * 3}f", *[coord for pos in points for coord in pos])
            msg = struct.pack(f"{len(points) * 2}f", *[coord for pos in points for coord in pos[:2]])
            msg += struct.pack(f"{len(cls)}i", *cls)
            print(f"Sending {len(points)} points and {len(cls)} class IDs")
            socket.send(msg)

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
