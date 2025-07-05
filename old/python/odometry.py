from ultralytics import YOLO
import cv2
import struct
import numpy as np
import argparse as ap
from ultralytics.utils.plotting import Annotator
import pyrealsense2 as rs
import zmq
import os
import time
import matplotlib.pyplot as plt
import json
import matplotlib

SHOW = False
REALTIME_TRAJECTORY = False
FRAMES_TO_DISCARD = 400

class_colors = {
    0: (255,   0,   0),
    1: (0,   140, 255),
    2: (0,   165, 255),
    3: (128, 128, 128),
    4: (0,   255, 255)
}

yaw = np.deg2rad(0)
cos_yaw = np.around(np.cos(yaw), decimals=5)
sin_yaw = np.around(np.sin(yaw), decimals=5)
x_translation = - 0.275 + 0.00625
y_translation = 0
z_translation = 0.747

transformation_matrix = np.array([
    [-sin_yaw, 0, cos_yaw, x_translation],
    [cos_yaw,  0, sin_yaw, y_translation],
    [0,        1,       0, z_translation],
    [0,        0,       0,             1]
])

parser = ap.ArgumentParser(
    description = 'A program used to run Visual Odometry with various Feature detection algorithms.'
)

parser.add_argument(
    '--extraction', '-e',
    type = str,
    default = "orb",
    choices = ['orb', 'sift', 'surf', 'akaze'],
    help = 'The feature extraction algorithm (defaults to ORB).')

parser.add_argument(
    '--matching', '-m',
    type = str,
    default = 'bf',
    choices = ['bf', 'flann'],
    help = 'The feature matching algorithm (defaults to BruteForce).'
)

parser.add_argument(
    '--realtime', '-rt',
    action = 'store_true',
    help = 'Enable Real Time acquisition from RealSense camera.'
)

parser.add_argument(
    '--path', '-p',
    type = str,
    help = 'If real-time is turned off, the argument locating the rosbag to replay is required.'
)

parser.add_argument(
    '--yolo', '-y',
    action = 'store_true',
    help = 'Choose whether to run the YOLO inference.'
)

parser.add_argument(
    '--pose', '-po',
    type = str,
    default = 'pnp',
    choices = ['pnp', 'emat'],
    help = 'Choose which method ought to be used to determine the pose, with scaled values or not.'
)

parser.add_argument(
    '--baseline', '-b',
    action = 'store_true',
    help = 'Enable baseline smth smth smth smth smth.'
)

parser.add_argument(
    '--basepath', '-bp',
    type = str,
    help = 'If baseline is turned on, the path to the baseline .json needs to be provided'
)

args = parser.parse_args()

if args.extraction == 'orb':
    extractor = cv2.ORB_create(3000)
elif args.extraction == 'sift':
    extractor = cv2.SIFT_create()
elif args.extraction == 'surf':
    extractor = cv2.xfeatures2d.SURF_create()
    extractor.setUpright(True)
    extractor.setExtended(True)
elif args.extraction == 'akaze':
    extractor = cv2.AKAZE_create()

if args.matching == 'bf':
    if args.extraction == 'orb' or args.extraction == 'akaze':
        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )
    elif args.extraction in ['sift', 'surf']:
        matcher = cv2.BFMatcher(
            cv2.NORM_L2,
            crossCheck=True
        )
elif args.matching == 'flann':
    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        
if args.yolo:
    model = YOLO('./best_seg.onnx', task="segment")


def init():
    pipeline = rs.pipeline()
    config = rs.config()
    pointcloud = rs.pointcloud()
    
    if args.realtime:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        profile = pipeline.start(config)
    else:
        if args.path is None:
            raise ValueError('--realtime was set to false but the PATH of the rosbg to replay wasn\'t provided!\n Use --path PATH or -p PATH to specify the location.')
        config.enable_device_from_file(os.path.expanduser(args.path), repeat_playback=False)
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(True)

        if args.baseline:
            if args.basepath is None:
                raise ValueError('Baseline was turned on but no path to the baseline json was provided.')
            baseline_path = os.path.expanduser(args.basepath)
            x_baseline, y_baseline = get_baseline(baseline_path)
    
    intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics
    align = rs.align(rs.stream.color)

    socket = zmq.Context().socket(zmq.PUB)
    socket.bind("tcp://*:5555")

    if args.realtime:
        return pipeline, align, intrinsics, pointcloud, socket
    else:
        if args.baseline:
            return pipeline, align, intrinsics, pointcloud, socket, x_baseline, y_baseline
        else:
            return pipeline, align, intrinsics, pointcloud, socket

def get_frames(pipeline, align, pc):
    try:
        frames = pipeline.wait_for_frames()
    except:
        print("------------------------------------------------")
        print("> Rosbag ended")
        exit()
    aligned = align.process(frames)
    depth_frame = aligned.get_depth_frame()
    color_frame = aligned.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    
    pc.map_to(color_frame)
    pointcloud = pc.calculate(depth_frame) 
    
    vertices = np.asarray(pointcloud.get_vertices())

    return depth_frame, depth_image, color_image, vertices

def compute_bboxes_masks(color_image, boxes, masks, model_names):
    annotator = Annotator(color_image)
    centroids = []
    classes = []
    for box, mask in zip(boxes, masks):
        b = box.xyxy[0]
        c = box.cls
        color = class_colors[int(c)]

        annotator.box_label(b, model_names[int(c)])

        if masks is not None and hasattr(masks, 'xy') and len(masks.xy) > 0 and len(mask.xy[0]) >= 3:
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
        else:
            continue
        
        
    return annotator.result(), centroids, classes

def cam_to_com(point):
    homogeneous_point = np.append(list(point),1)
    return np.matmul(transformation_matrix,homogeneous_point)[:3]

def detect_features(gray_img, mask):
    """Detect and compute chosen features."""
    return extractor.detectAndCompute(gray_img, mask)

def match_features(des1, des2):
    """Feature matching using chosen method."""
    matches = matcher.match(des1, des2)
    return sorted(matches, key = lambda x:x.distance)

def infere(img, depth_image, vertices):
    res = model.predict(img, imgsz=(480, 640))
    model_names = model.names
    points = []
    cls = []
    
    for r in res:
        print(f"Found {len(r.boxes)} boxes")
        if r.boxes.shape[0] != 0:
            _, centroids, classes = compute_bboxes_masks(img, r.boxes, r.masks, model_names)
            
            for i in range(len(centroids)):
                cx, cy = centroids[i]
                
                idx = cy * depth_image.shape[1] + cx
                point = vertices[idx]
                if SHOW:
                    cv2.circle(depth_image, (cx, cy), 6, (255, 255, 255), -1)
                print(f"Point - cam RF: {point}")
                com_point = cam_to_com(point)
                print(f"Point - com RF: {com_point}")
                points.append(com_point.tolist())
                cls.append(classes[i])
                print(f"Class ID: {classes[i]}")
                
    return points, cls

def estimate_motion(prev_pts, curr_pts, K):
    if args.extraction == 'orb':
        prob = 0.9
        threshold = 1
    elif args.extraction in ['sift', 'surf']:
        prob = 0.7
        threshold = 0.1

    E, mask = cv2.findEssentialMat(prev_pts, curr_pts, cameraMatrix=K, method=cv2.RANSAC, prob=prob, threshold=threshold)
    points_used, R, T, mask_RP = cv2.recoverPose(E, prev_pts, curr_pts, cameraMatrix=K, mask=mask)

    return R, T, points_used, mask_RP.flatten().astype(np.uint8)

def solve_pnp(matches, kp1, kp2, depth_frame, K):
    if not matches or len(kp1) == 0 or len(kp2) == 0:
        return None, None, None

    obj_points = []
    img_points = []

    depth_image = np.asanyarray(depth_frame.get_data())

    for m in matches:
        if m.queryIdx >= len(kp1) or m.trainIdx >= len(kp2):
            continue 
        
        u, v = map(int, kp1[m.queryIdx].pt)
        
        if u < 0 or v < 0 or u >= depth_image.shape[1] or v >= depth_image.shape[0]:
            continue
        
        depth = depth_image[v, u] * 0.001

        if depth > 0:
            x = (u - K[0, 2]) * depth / K[0, 0]
            y = (v - K[1, 2]) * depth / K[1, 1]
            obj_points.append((x, y, depth))
            img_points.append(kp2[m.trainIdx].pt)

    if len(obj_points) < 10:
        return None, None, None

    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)

    _, r, t, inliers = cv2.solvePnPRansac(obj_points, img_points, K, None)
    if inliers is None:
        return None, None, None
    R, _ = cv2.Rodrigues(r)
    
    return R, t, len(inliers)

def get_baseline(baseline_path):
    
    with open(baseline_path, 'r') as file:
        data = json.load(file)

    x_values = data['tracksBaseline']['Povo@DV_1']['x']
    y_values = data['tracksBaseline']['Povo@DV_1']['y']

    return x_values, y_values

def main():

    if args.realtime:
        pipeline, align, intrinsics, pointcloud, socket = init()

        #TODO: get real-time value from telemetry to define the roto-translation matrix for real-time usage
    else:
        if args.baseline:
            pipeline, align, intrinsics, pointcloud, socket, x_baseline, y_baseline = init()
        
            #compute roto-translation wrt telemetry baseline
            yaw = - np.pi/2 - np.arctan2(y_baseline[1] - y_baseline[0], x_baseline[1] - x_baseline[0])
            cos_yaw = np.around(np.cos(yaw), decimals=5)
            sin_yaw = np.around(np.sin(yaw), decimals=5)
            x_translation = x_baseline[0]
            y_translation = 0
            z_translation = y_baseline[0]

            transformation_matrix = np.array([
                [-cos_yaw,   0,   -sin_yaw,  x_translation],  # Negate X
                [       0,   1,         0,   y_translation],  # Y unchanged
                [ sin_yaw,   0,   -cos_yaw,  z_translation],  # Negate Z
                [       0,   0,         0,               1]
            ])
            
        else:
            pipeline, align, intrinsics, pointcloud, socket = init()
        
    
    if args.yolo:
        model = YOLO('./best_seg.onnx', task="segment")

    kp_old = None
    des_old = None

    fx = intrinsics.fx  
    fy = intrinsics.fy  
    cx = intrinsics.ppx  
    cy = intrinsics.ppy 

    K = np.array([
              [fx,  0, cx],
              [ 0, fy, cy],
              [ 0,  0,  1]
            ])

    trajectory_x = []
    trajectory_y = []
    trajectory_z = []
    #TODO: unificare trajectory points

    if args.baseline:
        roto_translation_x = []
        roto_translation_y = []

    avg_time = []
    start = time.time()

    count = 0
    poses = [np.eye(4, dtype=np.float64)]
    
    # What we could do is: instead of a triangle mask, we could try to create a custom one for the car,
    # so it works however the camera is oriented. To so so we could try to separete the forground (which should be
    # the car), from the background. I'll try this later
    _, _, img, _ = get_frames(pipeline, align, pointcloud)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    h, w = img.shape[:2]
    pt1 = (w//2, h*2//3-10)
    pt2 = (0+20, h)
    pt3 = (w-20, h)
    triangle_cnt = np.array([pt1, pt2, pt3])
    cv2.drawContours(mask, [triangle_cnt], 0, (255), thickness=-1)
    mask = cv2.bitwise_not(mask)

    matplotlib.use('TkAgg')
    if REALTIME_TRAJECTORY:
        plt.figure()
        plt.axis('equal')
        scatter = plt.scatter([], [], c='blue')
        if not args.realtime and args.baseline:
            plt.scatter(x_baseline, y_baseline, c='red')
            plt.ylabel('Y-axis')
            plt.title('Estimated trajectory - Baseline RF')

        else:
            plt.xlabel('X-axis')
            plt.ylabel('Z-axis')
            plt.title('Estimated trajectory - Camera RF')
        plt.ion()
        plt.show()

        min_x, max_x = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')

    try:
        while True:

            depth_frame, depth_image, color_image, vertices = get_frames(pipeline, align, pointcloud)
            
            count = count + 1
            print("# Frame: ", count)

            if color_image is None or depth_image is None:
                continue
            
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            if args.yolo:
                points, cls = infere(color_image, depth_image, vertices)

                msg = struct.pack(f"{len(points) * 3}f", *[coord for pos in points for coord in pos])
                msg = struct.pack(f"{len(points) * 2}f", *[coord for pos in points for coord in pos[:2]])
                msg += struct.pack(f"{len(cls)}i", *cls)
                print(f"Sending {len(points)} points and {len(cls)} class IDs")
                socket.send(msg)

            gray_img = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
            kp_new, des_new = detect_features(gray_img, mask)

            if kp_old is not None and des_old is not None:

                start = time.time()

                matches = match_features(des_old, des_new)

                if args.pose == 'pnp':
                    R, T, points_used = solve_pnp(matches, kp_old, kp_new, depth_frame, K)
                else:   
                    src_pts = np.float32([kp_old[m.queryIdx].pt for m in matches])
                    dst_pts = np.float32([kp_new[m.trainIdx].pt for m in matches])  # Get the corresponding points in the target image
 
                    R, T, points_used, pose_mask = estimate_motion(src_pts, dst_pts, K)   

                if SHOW:
                    if args.pose == 'pnp':
                        N = len(kp_old)*2//3
                        img_matching = cv2.drawMatches(gray_img_old,kp_old,gray_img,kp_new,matches[:N],None,matchColor=(0, 255, 0), flags=2)
                        cv2.imshow('First 2/3 of feature matches', img_matching)
                    else:
                        img_matching = cv2.drawMatches(gray_img_old,kp_old,gray_img,kp_new,matches[:],None,matchColor=(0, 255, 0), matchesMask=pose_mask.ravel().tolist(), flags=2)
                        cv2.imshow('Feature matching used to compute position', img_matching)             

                if points_used is not None and points_used > 15:

                    # R and T are relative between first and second frame, not in global unit

                    T_prev = poses[-1]
                    T_rel = np.eye(4, dtype=np.float64)
                    T_rel[0:3, 0:3] = R
                    T_rel[0:3, 3] = T.flatten()

                    T_curr = T_prev @ T_rel
                    poses.append(T_curr)

                    trajectory_x.append(T_curr[0, 3])
                    trajectory_y.append(T_curr[1, 3])
                    trajectory_z.append(-T_curr[2, 3])

                    if args.baseline:
                        trajectory_telemetry = np.eye(4, dtype=np.float64)
                        trajectory_telemetry[0:3, 3] = [T_curr[0, 3], T_curr[1, 3], -T_curr[2, 3]]
                        new = transformation_matrix @ trajectory_telemetry
                        roto_translation_x.append(new[0, 3])
                        roto_translation_y.append(new[2,3])

                    avg_time.append(time.time() - start)

                    if REALTIME_TRAJECTORY:
                        if args.baseline:
                            padding = 5
                            min_x = min(min_x, new[0,3] - padding)
                            max_x = max(max_x, new[0,3] + padding)
                            min_z = min(min_z, new[2,3] - padding)
                            max_z = max(max_z, new[2,3] + padding)
                            
                            scatter.set_offsets(np.column_stack((roto_translation_x, roto_translation_y)))
                            axis_range = max(max_x - min_x, max_z - min_z) 
                            mid_x = (min_x + max_x) / 2
                            mid_z = (min_z + max_z) / 2 
                            plt.xlim(mid_x - axis_range / 2, mid_x + axis_range / 2)
                            plt.ylim(mid_z - axis_range / 2, mid_z + axis_range / 2)
                            plt.gca().set_aspect('equal', adjustable='box')

                            plt.draw()
                            plt.pause(0.1)

                        else:
                            padding = 5
                            min_x = min(min_x, T_curr[0, 3] - padding)
                            max_x = max(max_x, T_curr[0, 3] + padding)
                            min_z = min(min_z, -T_curr[2, 3] - padding)
                            max_z = max(max_z, -T_curr[2, 3] + padding)
                            
                            scatter.set_offsets(np.column_stack((trajectory_x, trajectory_z)))
                            axis_range = max(max_x - min_x, max_z - min_z) 
                            mid_x = (min_x + max_x) / 2
                            mid_z = (min_z + max_z) / 2 
                            plt.xlim(mid_x - axis_range / 2, mid_x + axis_range / 2)
                            plt.ylim(mid_z - axis_range / 2, mid_z + axis_range / 2)
                            plt.gca().set_aspect('equal', adjustable='box')

                            plt.draw()
                            plt.pause(0.1)

                else:
                    kp_new = kp_old
                    des_new = des_old
                    gray_img = gray_img_old

            kp_old = kp_new
            des_old = des_new
            gray_img_old = gray_img


            if SHOW:
                depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                images = np.hstack((color_image, depth_image))

                cv2.imshow('YOLO V8 Segmentation', images)

                if cv2.waitKey(1) & 0xFF == ord(' '):
                    cv2.destroyAllWindows()
                    break
                

    finally:
        print("> Pipe stop\n")
        pipeline.stop()

        vo_avgTime = 0
        for t in avg_time:
            vo_avgTime += t
        vo_avgTime /= len(avg_time)
        print("VO - avg time per frame: ", vo_avgTime)
        print("------------------------------------------------")

        if REALTIME_TRAJECTORY:
            plt.ioff()
            plt.show()
        else:
            plt.figure()
            plt.axis('equal')
            if not args.realtime and args.baseline:
                plt.scatter(roto_translation_x, roto_translation_y, c='blue')
                plt.scatter(x_baseline, y_baseline, c='red')
                plt.ylabel('Y-axis')
                plt.title('Estimated trajectory - Baseline RF')
            else:
                plt.scatter(trajectory_x, trajectory_z, c='blue')
                plt.xlabel('X-axis')
                plt.ylabel('Z-axis')
                plt.title('Estimated trajectory - Camera RF')
            plt.show()

if __name__ == "__main__":
    main()