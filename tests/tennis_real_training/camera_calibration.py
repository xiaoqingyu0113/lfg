
import cv2
import numpy as np  
import yaml


ARUCO_DICT = {
  "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
  "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
  "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
  "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
  "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
  "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
  "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
  "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
  "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
  "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
  "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
  "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
  "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
  "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
  "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
  "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
  "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
  "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def detect_corners(img, selected_aruco_key):
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT.get(selected_aruco_key))
    parameters = cv2.aruco.DetectorParameters()
    corners, ids, rejected = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)
    img = cv2.aruco.drawDetectedMarkers(img, corners, ids)
    return img, corners, ids, rejected

if __name__ == '__main__':
    selected_aruco_key = "DICT_6X6_50"
    date = 'Dec13'
    filename_ = 'conf/camera/23045009_calibration'
    marker_length = 1.0404 # meters

    # read intrinsics from old yaml file
    with open(f'{filename_}_Jul18.yaml') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
        # distortion_coefficients = np.array(data['distortion_coefficients']['data'])
        distortion_coefficients = None

    # detect corners
    img = cv2.imread(f'{filename_}_{date}.jpg')
    img, corners, ids, rejected = detect_corners(img, selected_aruco_key)

    if len(ids) == 0:
        raise ValueError("No markers found in the image")
    elif len(ids) > 1:
        raise ValueError("Multiple markers found in the image")
    
    # estimate pose
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners[0], marker_length, camera_matrix, distortion_coefficients)
    cv2.drawFrameAxes(img, camera_matrix, distortion_coefficients, rvec, tvec, 2.0)
    
    # convert rvec to rotation matrix
    rotm_cam_world = cv2.Rodrigues(rvec)[0]
    t_cam_world = tvec[0,0].reshape(3,1)
    t_world_cam = -rotm_cam_world.T @ t_cam_world
    
    # # transform to world frame
    if '22495525' in filename_ or '22495526' in filename_ or '22495527' in filename_:
        rotm_offset = np.eye(3)
        # translation after rotation offeset, should include court poster edge to tag edge, line width, tag width, 
        t_offset_world_cam = np.array([0.520+0.020+0.025+5.48, -0.520-0.020-0.025, 0.0]).reshape(3,1)

        rotm_new_cam_world =  rotm_cam_world
        t_new_world_cam = t_world_cam + t_offset_world_cam
        t_new_cam_world = -rotm_new_cam_world @ t_new_world_cam

        print(f't_world,cam', t_new_world_cam)
        print('R_cam_world', rotm_cam_world)
    

    elif '23045007' in filename_ or '23045008' in filename_ or '23045009' in filename_:
        # rotate about z axis by 180
        rotm_offset = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
        # translation after rotation offeset, should include court poster edge to tag edge, line width, tag width, 
        t_offset_world_cam = np.array([23.77-0.520-0.020-0.025-5.48, 0.520+0.020+0.025, 0.0]).reshape(3,1) # after rotation

        rotm_new_cam_world =   (rotm_offset @ rotm_cam_world.T).T
        t_new_world_cam = rotm_offset @ t_world_cam + t_offset_world_cam
        t_new_cam_world = -rotm_new_cam_world @ t_new_world_cam

        print(f't_world,cam', t_new_world_cam)
        print('R_cam_world', rotm_cam_world)

    # save new calibration
    with open(f'{filename_}_{date}.yaml', 'w') as file:
        yaml.dump({'camera_matrix': {'rows': 3, 'cols': 3, 'data': camera_matrix.flatten().tolist()},
                    'R_cam_world': rotm_new_cam_world.flatten().tolist(),
                    't_world_cam': t_new_world_cam.flatten().tolist(),
                    'distorsion_coefficients': {'rows': 1, 'cols': 5, 'data': data['distortion_coefficients']['data']}}, file)
    print(f'New calibration saved to {filename_}_{date}.yaml')

    # # read intrinsics from old yaml file
    # with open(f'{filename_}_Jul18.yaml') as file:
    #     data = yaml.load(file, Loader=yaml.FullLoader)
    #     camera_matrix = np.array(data['camera_matrix']['data']).reshape(3, 3)
    #     rotm_new_cam_world = np.array(data['R_cam_world']).reshape(3, 3)
    #     t_new_world_cam = np.array(data['t_world_cam'])
    #     t_new_cam_world = -rotm_new_cam_world @ t_new_world_cam

    # project court lines
    rvec_new = cv2.Rodrigues(rotm_new_cam_world)[0].reshape(1,1,3)
    tvec_new = t_new_cam_world.reshape(1,1,3)
    court_lines = {
        "baseline": [[0, -5.48, 0], [0, 5.48, 0]],             # Bottom baseline
        "service_line": [[5.48, -4.11, 0], [5.48, 4.11, 0]],     # Bottom service line
        "lside_line": [[0, -5.48, 0], [11.88, -5.48, 0]],         # Left side line
        "rside_line": [[0, 5.48, 0], [11.88, 5.48, 0]],           # Right side line
        "lserve_line_inner": [[0, -4.11, 0], [11.88, -4.11, 0]],   # Left service line inner
        "rserve_line_inner": [[0, 4.11, 0], [11.88, 4.11, 0]],     # Right service line inner
        "center_service_line": [[5.48, 0, 0], [11.8, 0, 0]],   # Left service line outer
    }

    # Plot each court line
    for name, points in court_lines.items():
        # Convert 3D points to 2D using projectPoints
        line_3d = np.array(points, dtype=float).reshape(1, 2, 3)
        imgpts, _ = cv2.projectPoints(line_3d, rvec_new, tvec_new, camera_matrix, distortion_coefficients)

        # Draw the line on the image
        img = cv2.line(
            img,
            tuple(imgpts[0, 0].astype(int)),
            tuple(imgpts[1, 0].astype(int)),
            (0, 255, 255),  # Yellow color
            2  # Line thickness
        )

        # mirrow 
        imgpts, _ = cv2.projectPoints(np.array([23.77, 0, 0]) - line_3d, rvec_new, tvec_new, camera_matrix, distortion_coefficients)

        # Draw the line on the image
        img = cv2.line(
            img,
            tuple(imgpts[0, 0].astype(int)),
            tuple(imgpts[1, 0].astype(int)),
            (0, 255, 255),  # Yellow color
            2  # Line thickness
        )


    # save image
    cv2.imwrite(f'{filename_}_{date}_pose.jpg', img)
    print(f'Image with pose saved to {filename_}_{date}_pose.jpg')