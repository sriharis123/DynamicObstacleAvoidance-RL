from matplotlib.offsetbox import OffsetImage
import numpy as np
import matplotlib.pylab as pl
import yaml
import scipy.ndimage as ndimage
from obstacle import Obstacle

#from moviepy.editor import VideoClip #moviepy v. 0.2.2.11
car_image = pl.imread('config/car.png', format='png')
def get_image(angle):
    return OffsetImage(ndimage.rotate(car_image, angle, reshape=True), zoom=0.3)

def connect_segments(segments, resolution = 0.01):
    """
    :param segments: start and end points of all segments as ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
           step_size : resolution for plotting
    :return: stack of all connected line segments as (X, Y)
    """

    for i, seg_i in enumerate(segments):
        if seg_i[1] == seg_i[3]: #horizontal segment
            x = np.arange(min(seg_i[0],seg_i[2]), max(seg_i[0],seg_i[2]), resolution)
            y = seg_i[1]*np.ones(len(x))
        elif seg_i[0] == seg_i[2]: #vertical segment
            y = np.arange(min(seg_i[1],seg_i[3]), max(seg_i[1],seg_i[3]), resolution)
            x = seg_i[0]*np.ones(len(y))
        else: # gradient exists
            m = (seg_i[3] - seg_i[1])/(seg_i[2] - seg_i[0])
            c = seg_i[1] - m*seg_i[0]
            x = np.arange(min(seg_i[0],seg_i[2]), max(seg_i[0],seg_i[2]), resolution)
            y = m*x + c

        obs = np.vstack((x, y)).T
        if i == 0:
            connected_segments = obs
        else:
            connected_segments = np.vstack((connected_segments, obs))

    return connected_segments

def get_filled_txy(dist_theta, robot_pos, fov, n_reflections, max_laser_distance, unoccupied_points_per_meter=0.1, margin=0.1):
    """
    :param dist_theta: lidar hit distance
    :param robot_pos: robot pose
    :param fov: robot field of view
    :param n_reflections: number of lidar hits
    :param max_laser_distance: maximum lidar distance
    :param unoccupied_points_per_meter: in-fill density
    :param margin: in-fill density of free points
    :return: (points, labels) - 0 label for free points and 1 label for hits
    """

    angles = np.linspace(robot_pos[2], robot_pos[2] + fov, n_reflections)
    laser_data_xy = np.vstack([dist_theta * np.cos(angles), dist_theta * np.sin(angles)]).T + robot_pos[:2]

    for i, ang in enumerate(angles):
        dist = dist_theta[i]
        laser_endpoint = laser_data_xy[i,:]

        # parametric filling
        para = np.sort(np.random.random(np.int16(dist * unoccupied_points_per_meter)) * (1 - 2 * margin) + margin)[:, np.newaxis]  # TODO: Uniform[0.05, 0.95]
        points_scan_i = robot_pos[:2] + para*(laser_endpoint - robot_pos[:2])  # y = <x0, y0, z0> + para <x, y, z>; para \in [0, 1]

        if i == 0:  # first data point
            if dist >= max_laser_distance:  # there's no laser reflection
                points = points_scan_i
                labels = np.zeros((points_scan_i.shape[0], 1))
            else:  # append the arrays with laser end-point
                points = np.vstack((points_scan_i, laser_endpoint))
                labels = np.vstack((np.zeros((points_scan_i.shape[0], 1)), np.array([1])[:, np.newaxis]))
        else:
            if dist >= max_laser_distance:  # there's no laser reflection
                points = np.vstack((points, points_scan_i))
                labels = np.vstack((labels, np.zeros((points_scan_i.shape[0], 1))))
            else:  # append the arrays with laser end-point
                points = np.vstack((points, np.vstack((points_scan_i, laser_endpoint))))
                labels = np.vstack((labels, np.vstack((np.zeros((points_scan_i.shape[0], 1)), np.array([1])[:, np.newaxis]))))

    #pl.scatter(points[:,0], points[:,1], c=labels, s=10)
    #pl.axis('equal')
    #pl.show()
    #sys.exit()
    return np.hstack((points, labels))

def load_obstacles_config(environment):
    """
    :param environment: name of the yaml config file
    :return: all obstacles, area of the environment
    """
    with open('config/'+environment+'.yaml') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)

        # load environment area parameters
        area = yaml_data['area']
        area = (area['x_min'], area['x_max'], area['y_min'], area['y_max'])

        # load static and dynamic obstacles
        obs = yaml_data['obstacles']
        all_obstacles = []
        for i in range(len(obs)):
            obs_i = Obstacle(centroid=[obs[i]['centroid_x'], obs[i]['centroid_y']], dx=obs[i]['dx'], dy=obs[i]['dy'],
                            angle=obs[i]['orientation']*np.pi/180, vel=[obs[i]['velocity_x'], obs[i]['velocity_y']],
                            acc=[obs[i]['acc_x'], obs[i]['acc_y']], area=area)
            all_obstacles.append(obs_i)
    return all_obstacles, area

def load_robot_config(environment):
    with open('config/'+environment+'.yaml') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)

        # load environment area parameters
        robot = yaml_data['robot']
        centroid = (robot['centroid_x'], robot['centroid_y'])
        orientation = robot['orientation']
        velocity = robot['velocity']

    return centroid, orientation, velocity

def load_goal_config(environment):
    with open('config/'+environment+'.yaml') as file:
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        goal = yaml_data['goal']

    return (goal['x_loc'], goal['y_loc'])
