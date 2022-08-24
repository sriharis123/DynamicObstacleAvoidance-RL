from animation_util import load_obstacles_config, connect_segments, load_goal_config, load_robot_config
import torch
import numpy as np
import matplotlib.pylab as pl
from car import Car

def get_laser_ref_covered(segments, fov=np.pi, n_reflections=180, max_dist=100, xytheta_robot=np.array([0.0, 0.0])):
    """
    :param segments: start and end points of all segments as ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
           fov: sight of the robot - typically pi or 4/3*pi
           n_reflections: resolution=fov/n_reflections
           max_dist: max distance the robot can see. If no obstacle, laser end point = max_dist
           xy_robot: robot's position in the global coordinate system
    :return: 1xn_reflections array indicating the laser end point
    """

    covered_segments = segments[:2]
    sys.exit()

    xy_robot = xytheta_robot[:2] #robot position
    theta_robot = xytheta_robot[2] #robot angle in rad

    angles = np.linspace(theta_robot, theta_robot+fov, n_reflections)
    dist_theta = max_dist*np.ones(n_reflections) # set all laser reflections to 100

    for seg_i in segments:
        xy_i_start, xy_i_end = np.array(seg_i[:2]), np.array(seg_i[2:]) #starting and ending points of each segment
        for j, theta in enumerate(angles):
            xy_ij_max = xy_robot + np.array([max_dist*np.cos(theta), max_dist*np.sin(theta)]) # max possible distance
            intersection = get_intersection(xy_i_start, xy_i_end, xy_robot, xy_ij_max)

            if intersection is not None: #if the line segments intersect
                r = np.sqrt(np.sum((intersection-xy_robot)**2)) #radius

                if r < dist_theta[j]:
                    dist_theta[j] = r

    return dist_theta

def get_way_points_gui(environment, vehicle_poses=None):
    """
    :param environment: yaml config file
    :param vehicle_poses: vehicle poses
    :return:
    """
    class mouse_events:
        def __init__(self, fig, line):
            self.path_start = False #If true, capture data
            self.fig = fig
            self.line = line
            self.xs = list(line.get_xdata())
            self.ys = list(line.get_ydata())
            self.orientation = []

        def connect(self):
            self.a = self.fig.canvas.mpl_connect('button_press_event', self.__on_press)
            self.b = self.fig.canvas.mpl_connect('motion_notify_event', self.__on_motion)

        def __on_press(self, event):
            print('You pressed', event.button, event.xdata, event.ydata)
            self.path_start = not self.path_start

        def __on_motion(self, event):
            if self.path_start is True:
                if len(self.orientation) == 0:
                    self.orientation.append(0)
                else:
                    self.orientation.append(np.pi/2 + np.arctan2((self.ys[-1] - event.ydata), (self.xs[-1] - event.xdata)))
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                self.line.set_data(self.xs, self.ys)
                self.line.figure.canvas.draw()

    # set up the environment
    all_obstacles, area = load_obstacles_config(environment=environment)

    # update obstacles
    all_obstacle_segments = []
    for obs_i in all_obstacles:
        all_obstacle_segments += obs_i.update()

    connected_components = connect_segments(all_obstacle_segments)

    # plot
    pl.close('all')
    fig = pl.figure()#figsize=(10, 5))  # (9,5)
    ax = fig.add_subplot(111)
    pl.title('Generate waypoints: 1) Click to start. 2) Move the mouse. \n3) Click to stop. 4) lose the gui to exit')
    ax.scatter(connected_components[:, 0], connected_components[:, 1], marker='.', c='y', edgecolor=['none'], alpha=0.2)  # obstacles
    if vehicle_poses is not None:
        pl.plot(vehicle_poses[:, 0], vehicle_poses[:, 1], 'o--', c='m')
    pl.xlim(area[:2]); pl.ylim(area[2:])

    line, = ax.plot([], [])
    mouse = mouse_events(fig, line)
    mouse.connect()

    pl.show()

    return np.hstack((np.array(mouse.xs)[:, None], np.array(mouse.ys)[:, None], np.array(mouse.orientation)[:,None]))[1:]

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

    return np.hstack((points, labels))

def rollout(lstm, env='static_180', rolloutLimit=200, isTest=False):

    # robot LIDAR configuration
    n_reflections = 18 # number of lidar beams in the 2D plane
    fov = 360 *np.pi/180 # lidar field of view in degrees
    max_laser_distance = 15 # maximum lidar distance in meters
    unoccupied_points_per_meter = 0.5 # beam filling configuration

    all_obstacles, area = load_obstacles_config(environment=env)
    r_centroid, r_orientation, r_vel = load_robot_config(environment=env)
    goal_loc = load_goal_config(environment=env)


    # create car agent
    x_init        = r_centroid[0]#np.random.randint(0,80) + 10
    y_init        = r_centroid[1]#np.random.randint(0,30) + 10
    L             = 2
    theta_init    = r_orientation * np.pi / 180     # in radians
    phi_init      = 0
    velocity_init = r_vel
    dt           = 0.1  # s
    goal_loc     = [goal_loc[0], goal_loc[1]]
    max_velocity = 20
    min_velocity = -6

    car = Car(x_init, y_init, L, theta_init, phi_init, velocity_init, dt, goal_loc, max_velocity, min_velocity, n_reflections=n_reflections, fov=fov, max_laser_distance=max_laser_distance)
    
    phi_dot = 0
    v_dot = car.max_v_dot

    # set up initial lidar reading
    # update obstacles
    all_obstacle_segments = []
    for obs_i in all_obstacles:
        all_obstacle_segments += obs_i.update(0)

    # get the new lidar points
    car.update(0, 0, all_obstacle_segments)

    tau = []
    obs = []
    measurements = car.state_numpy()
    inputs = []
    outputs = []

    for t in range(rolloutLimit):
        
        input_tensor = None
        if t < 5:
            input_tensor = torch.empty((1, t+1, len(car.dist_theta) + 5))
            input_tensor[0, :, :] = torch.tensor(measurements)
        else:
            input_tensor = torch.empty((1, 5, len(car.dist_theta) + 5))
            input_tensor[0, :, :] = torch.tensor(measurements[t-4:])

        inputs.append(input_tensor)
        probs = lstm(input_tensor)

        L = 0.5
        R = 0.5
        if not isTest:
            if (torch.rand(1) < torch.min(probs)):
                outputs.append(torch.argmin(probs[0,0]))
                L = probs[0,0,1].item()
                R = probs[0,0,0].item()
            else:
                outputs.append(torch.argmax(probs[0,0]))
                L = probs[0,0,0].item()
                R = probs[0,0,1].item()
        else:
            L = probs[0,0,0].item()
            R = probs[0,0,1].item()
            # determine actions to take
        v_dot = v_dot # no network for velocity right now
        phi_dot = L*car.max_phi_dot*dt - R*car.max_phi_dot*dt

        # update obstacles. TODO: fix Obstacle update.
        segs = []
        for obs_i in all_obstacles:
            segs += obs_i.update(car.dt)

        # update car with kinematics and obstacles
        car.update(v_dot, phi_dot, segs)

        # determine reward and whether rollout is complete
        reward, end = car.reward()
        
        # high negative reward if goal is not reached at end of rollout
        if t == rolloutLimit - 1 and not car.goal_reached:
            reward = -10
        
        # append step info
        tau.append((car.s, car.a, car.sp, reward))
        obs.append(segs)
        measurements = np.vstack((measurements, car.state_numpy()))

        # if game over end rollout
        if end:
            break

    if isTest:
        return (tau, obs, area, car)
    return (tau, obs, area, car, inputs, outputs)