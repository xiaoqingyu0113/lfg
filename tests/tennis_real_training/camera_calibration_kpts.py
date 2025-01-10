import numpy as np
import cv2
from scipy.optimize import least_squares
import yaml
from typing import List, Dict
from dataclasses import dataclass, field
import os 

@dataclass
class KeyPoints:
    selected_point: List[int] = field(default_factory=list)
    point_type: str = 'none'
    all_points: Dict[str, List[int]] = field(default_factory=dict)

    def update(self):
        self.all_points.update({self.point_type: self.selected_point})

def select_points(img='conf/camera/22495525_calibration_Dec13.jpg'):

    names = img.split('.')[0].split('/')[-1].split('_')
    serial, _, data = names
    keypoints_filename = f'conf/camera/{serial}_courtpoints_{data}.yaml'

    if os.path.exists(keypoints_filename):
        print('keypoints already exists. Load the file to update the points')
        keypoints_dict = yaml.load(open(keypoints_filename), Loader=yaml.FullLoader)
        keypoints = KeyPoints(all_points=keypoints_dict)
    else:
        keypoints = KeyPoints()
    # draw a line by clicking two points on the image
    image = cv2.imread(img)
    
    def draw_cross(img, x, y, color=(0, 0, 255), size=5):
        cv2.line(img, (x - size, y), (x + size, y), color, 1)
        cv2.line(img, (x, y - size), (x, y + size), color, 1)
        cv2.circle(img, (x, y), 2, color, -1)
    

    def draw_points(event, x, y, flags, param):
        img_copy = image.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            keypoints.selected_point = [x, y]
            if keypoints.point_type != 'none':
                keypoints.update()
        if event == cv2.EVENT_MOUSEMOVE:
            # cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)
            draw_cross(img_copy, x, y)
            cv2.putText(img_copy, keypoints.point_type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
        if event == cv2.EVENT_RBUTTONDOWN:
            # remove the last point
            if keypoints.all_points:
                keypoints.all_points.pop(keypoints.point_type, None)

        # draw all points
        for k, v in keypoints.all_points.items():
            # cv2.circle(img_copy, (v[0], v[1]), 5, (255, 0, 0), 1)
            draw_cross(img_copy, v[0], v[1])
            cv2.putText(img_copy, k, (v[0], v[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('image', img_copy)

    cv2.imshow('image', image)
    cv2.setMouseCallback('image', draw_points)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif 32 <= key  <= 126:
            keypoints.point_type = chr(key)
        
    cv2.destroyAllWindows()

    #remove .yaml from the image name
    
    print('saved to ', keypoints_filename )
    with open(keypoints_filename, 'w') as f:
        yaml.dump(keypoints.all_points, f)


def get_points_world():
    return {
        'm': [0,18,0], # in feet
        'n': [0, 13.5, 0],
        'o': [0,0, 0],
        'p': [0, -13.5, 0],
        'q': [0, -18, 0],
        'j': [18, 13.5,0],
        'k': [18, 0, 0],
        'l': [18, -13.5, 0],
        'v': [60, 13.5, 0],
        'w': [60, 0, 0],
        'x': [60, -13.5, 0],
        'u': [78, 18, 0],
        't': [78, 13.5, 0],
        'y': [78, 0, 0],
        's': [78, -13.5, 0],
        'r': [78, -18, 0],
    }

# compute extrinsic parameters based on the selected points
def compute_extrinsic_parameters(all_points, K):
    '''
    save K, R, T to a yaml file
    '''
    
    world_point_dict = get_points_world()
    world_points = []
    image_points = []
    for k, v in all_points.items():
        image_points.append(v)
        world_points.append(world_point_dict[k])
    
    image_points = np.array(image_points).astype(np.float32)
    world_points = np.array(world_points).astype(np.float32) * 0.3048 # convert to meters

    image_points = image_points.reshape(-1, 2, 1)
    world_points = world_points.reshape(-1, 3, 1)
    distCoeffs = np.zeros((5,1))


    # compute the rotation and translation
    _, rvec, tvec = cv2.solvePnP(world_points, image_points, K, np.zeros((5,1), dtype=np.float32)
)
    R = cv2.Rodrigues(rvec)[0]
    T = - R.T @ tvec

    print('R', R)
    print('T', T)
    
    return R, T

def draw_courtlines(imgname, K, R, T):
    img = cv2.imread(imgname)
    T = -R @ T
    rvec_new = cv2.Rodrigues(R)[0].reshape(1,1,3)
    tvec_new = T.reshape(1,1,3)
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
        imgpts, _ = cv2.projectPoints(line_3d, rvec_new, tvec_new, K, None)

        # Draw the line on the image
        img = cv2.line(
            img,
            tuple(imgpts[0, 0].astype(int)),
            tuple(imgpts[1, 0].astype(int)),
            (0, 255, 255),  # Yellow color
            2  # Line thickness
        )

        # mirrow 
        imgpts, _ = cv2.projectPoints(np.array([23.77, 0, 0]) - line_3d, rvec_new, tvec_new, K, None)

        # Draw the line on the image
        img = cv2.line(
            img,
            tuple(imgpts[0, 0].astype(int)),
            tuple(imgpts[1, 0].astype(int)),
            (0, 255, 255),  # Yellow color
            2  # Line thickness
        )

    # cv2.imshow('image', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # save image
    serial, _, date = imgname.split('.')[0].split('/')[-1].split('_')
    filename_ = f'conf/camera'
    cv2.imwrite(f'{filename_}/{serial}_calibration_{date}_pose_kpts.jpg', img)
    print(f'{filename_}/{serial}_calibration_{date}_pose_kpts.jpg')


if __name__ == "__main__":


    folder = 'conf/camera'
    serial = '23045007'
    date = 'Dec13'

    img = f'{folder}/{serial}_calibration_{date}.jpg'
    select_points(img)
    params = yaml.load(open(f'{folder}/{serial}_calibration_{date}.yaml'), Loader=yaml.FullLoader)
    K = np.array(params['camera_matrix']['data']).reshape(3, 3).astype(np.float32)
    R, T = compute_extrinsic_parameters(yaml.load(open(f'{folder}/{serial}_courtpoints_{date}.yaml'), Loader=yaml.FullLoader), K)

    with open(f'{folder}/{serial}_calibration_{date}_pose_kpts.yaml', 'w') as f:
        new_params = {
            'camera_matrix': {'rows': 3, 'cols': 3, 'data': K.flatten().tolist()},
            'R_cam_world': R.flatten().tolist(),
            't_world_cam': T.flatten().tolist()
        }
        yaml.dump(new_params, f)

    
    draw_courtlines(img, K, R, T)