import numpy as np
import cv2
from scipy.optimize import least_squares
import yaml



def select_points(img='conf/camera/22495525_calibration_Dec13.jpg'):
    # draw a line by clicking two points on the image
    img = cv2.imread(img)
    points =[]

    def draw_line(event, x, y, flags, param):
        img_copy = img.copy()
        if len(points) == 1:
            cv2.circle(img_copy, (points[0][0], points[0][1]), 3, (0, 0, 255), -1)
            cv2.line(img_copy, (points[0][0], points[0][1]), (x, y), (0, 255, 0), 2)
        
        if len(points) >= 2:
            cv2.line(img_copy, (points[0][0], points[0][1]), (points[1][0], points[1][1]), (0, 255, 0), 2)
            cv2.circle(img_copy, (points[0][0], points[0][1]), 3, (0, 0, 255), -1)
            cv2.circle(img_copy, (points[1][0], points[1][1]), 3, (0, 0, 255), -1)
            # show points
            cv2.putText(img_copy, f'{points[0][0], points[0][1]}', (points[0][0], points[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img_copy, f'{points[1][0], points[1][1]}', (points[1][0], points[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            cv2.circle(img_copy, (x, y), 5, (0, 0, 255), -1)

        # hover
        if event == cv2.EVENT_MOUSEMOVE:
            # show current mouse point, don't save and draw history points
            if len(points) == 0:
                cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)

            if len(points) == 1:
                cv2.line(img_copy, (points[0][0], points[0][1]), (x, y), (0, 255, 0), 2)
                cv2.circle(img_copy, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('image', img_copy)

    cv2.imshow('image', img)
    cv2.setMouseCallback('image', draw_line)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(points)


def compute_extrinsics_from_lines(image_lines, world_lines, K):
    # import yaml
    # import numpy as np

    # with open('conf/camera/22495525_calibration_Dec13.yaml') as f:
    #     params = yaml.load(f, Loader=yaml.FullLoader)


    # # Intrinsic matrix
    # K = np.array(params['camera_matrix']['data']).reshape(3, 3)

    # # Observed 2D points for two lines
    # image_lines = [
    #     [[100, 200], [300, 400]],  # Line 1
    #     [[50, 150], [200, 250]]   # Line 2
    # ]

    # # Corresponding 3D world points for each line
    # world_lines = [
    #     [[0, 0, 0], [1, 1, 1]],  # Line 1
    #     [[1, 0, 0], [1, 1, 0]]   # Line 2
    # ]

    def compute_line_2d(p1, p2):
        # Compute 2D line in homogeneous coordinates
        return np.cross(p1, p2)

    def project_point(K, R, t, point3d):
        # Transform 3D point to camera coordinates
        p_cam = R @ np.array(point3d) + t
        # Project to 2D image
        p_img = K @ p_cam
        return p_img / p_img[2]  # Normalize

    def reprojection_error(params, world_lines, image_lines):
        rvec, tvec = params[:3], params[3:]
        R, _ = cv2.Rodrigues(rvec)
        t = np.array(tvec)

        error = []
        for i, world_line in enumerate(world_lines):
            # Project 3D points to 2D
            p1 = project_point(K, R, t, world_line[0])
            p2 = project_point(K, R, t, world_line[1])

            # Compute projected line
            proj_line = compute_line_2d(p1, p2)

            # Compute observed line
            img_line = compute_line_2d(
                np.array([*image_lines[i][0], 1]),
                np.array([*image_lines[i][1], 1])
            )

            # Compute error
            err = np.linalg.norm(proj_line - img_line)
            error.append(err)
        print("Error:", error)
        return np.array(error)

    # Initial guesses for R and t
    rvec_init = np.zeros(3)
    tvec_init = np.zeros(3)
    params_init = np.hstack((rvec_init, tvec_init))

    # Optimize
    result = least_squares(
        reprojection_error, params_init, 
        args=(world_lines, image_lines), 
        method='lm'
    )

    # Extract optimized R and t
    rvec_opt, tvec_opt = result.x[:3], result.x[3:]
    R_opt, _ = cv2.Rodrigues(rvec_opt)
    t_opt = np.array(tvec_opt)

    print("Optimized Rotation Matrix (R):\n", R_opt)
    print("Optimized Translation Vector (t):\n", t_opt)


if __name__ == "__main__":

    # select_points('conf/camera/22495525_calibration_Dec13.jpg')
    params = yaml.load(open('conf/camera/22495525_calibration_Dec13.yaml'), Loader=yaml.FullLoader)
    K = np.array(params['camera_matrix']['data']).reshape(3, 3)
    world_lines = [
        [[0,18,0],[78,18,0]], # left1
        [[18, 0, 0], [60, 0, 0]],  # Line 1
    ]

    image_lines = [
        [[3,587],[822,364]], # left1
        [[266, 846], [1085, 481]],  # Line 1
    ]

    compute_extrinsics_from_lines(image_lines, world_lines, K)