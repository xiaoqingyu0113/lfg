import numpy as np
import rosbag
import matplotlib.pyplot as plt



def msg_parser(msg):
    t = msg.header.stamp.to_sec()
    points = []
    for p in msg.points:
        points.append([p.x, p.y])
    return t, points

def view_points():
    detections = {f'camera_{i}': [] for i in range(1,7)}
    bag = rosbag.Bag('/home/qingyu/Downloads/20241209_tennis/data1_2024-12-09-17-32-32.bag')

    # read all detections
    for topic, msg, t in bag.read_messages():

        camera_id = topic.split('/')[1]
        for p in msg.points:
            detections[camera_id].append([msg.header.stamp.to_sec(), p.x, p.y])
    bag.close()

    # convert to numpy array for ploting
    for camera_id, points in detections.items():
        detections[camera_id] = np.array(points)

    # plot out the detections
    fig, axs = plt.subplots(3, 2, figsize=(12, 8))
    for i, (camera_id, points) in enumerate(detections.items()):
        ax = axs[i//2, i%2]
        ax.scatter(points[:,1], points[:,2], s=1)
        ax.set_title(camera_id)
        ax.set_xlim(0, 1280)
        ax.set_ylim(0, 1024)
        ax.invert_yaxis()
    plt.show()

def main():
    view_points()




if __name__ == '__main__':
    main()