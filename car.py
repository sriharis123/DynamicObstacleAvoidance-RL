import numpy as np
from copy import deepcopy

class Car:
    # x, y in meters
    # angle in radians
    # velocity in meters/sec
    # dt in seconds

    def __init__(self, x_init, y_init, L, theta_init, phi_init, velocity_init, dt, goal_loc, 
                max_velocity=25.0, min_velocity=-25.0, max_angle=45,
                fov=180*np.pi/180, n_reflections=1, max_laser_distance=15, unoccupied_points_per_meter=0.5):
        # position values
        self.x_pos = x_init
        self.y_pos = y_init
        self.L = L
        self.theta = theta_init
        self.phi = phi_init
        self.dt = dt
        self.goal_loc = goal_loc
        self.dist_to_goal = np.sqrt((x_init - goal_loc[0]) ** 2 + (y_init - goal_loc[1]) ** 2)

        # kinematic values
        self.velocity = velocity_init

        # constraints on dynamics
        self.max_velocity = max_velocity
        self.min_velocity = min_velocity
        self.max_turn_angle = max_angle * np.pi / 180
        

        self.max_phi_dot = 50 * np.pi / 180
        self.max_v_dot = 7

        # lidar data
        self.dist_theta = [0] * n_reflections
        self.fov = fov
        self.n_reflections = n_reflections
        self.max_laser_distance = max_laser_distance
        self.unoccupied_points_per_meter = unoccupied_points_per_meter

        angle_goal_car    = np.arctan2((goal_loc[1] - y_init), (goal_loc[0] - x_init))
        if angle_goal_car < 0:
            angle_goal_car += 2 * np.pi
        self.angle_to_goal = angle_goal_car - self.theta
        if self.angle_to_goal > np.pi:
            self.angle_to_goal -= 2 * np.pi

        # initialize state variables
        self.s = None
        self.a = None
        self.sp = None

        self.goal_reached = False
        self.map_max_dist = np.sqrt(25**2 + 50**2)

    def get_robot_pos(self):
        return [self.x_pos, self.y_pos, self.theta]

    def apply_car_kinematics(self, v_dot, phi_dot):
        # position updates
        self.x_pos    += np.cos(self.theta) * self.velocity * self.dt
        self.y_pos    += np.sin(self.theta) * self.velocity * self.dt
        self.theta    += 1.0 / self.L * np.tan(self.phi) * self.velocity * self.dt
        if self.theta > 2 * np.pi:
            self.theta -= 2 * np.pi
        elif self.theta < 0:
            self.theta += 2 * np.pi

        # velocity updates
        self.phi      = clamp(self.phi + phi_dot * self.dt, -self.max_turn_angle, self.max_turn_angle)
        self.velocity = clamp(self.velocity + v_dot * self.dt, self.min_velocity, self.max_velocity)

    def update(self, v_dot, phi_dot, obstacles):

        # save old state for reward calculation
        # contains lidar readings, goal dist, and goal angle at time t

        # self.dist_theta = get_laser_ref(obstacles, self.fov, self.n_reflections, self.max_laser_distance, self.get_robot_pos())

        self.s = (deepcopy(self.dist_theta), self.dist_to_goal, self.angle_to_goal * 180 / np.pi, self.get_robot_pos())

        # save old action taken
        self.a = (v_dot, phi_dot)

        # apply kinematics
        self.apply_car_kinematics(v_dot, phi_dot)

        # update dist to goal
        self.dist_to_goal = np.sqrt((self.x_pos - self.goal_loc[0]) ** 2 + (self.y_pos - self.goal_loc[1]) ** 2)

        # update angle to goal
        angle_goal_car    = np.arctan2((self.goal_loc[1] - self.y_pos), (self.goal_loc[0] - self.x_pos))
        if angle_goal_car < 0:
            angle_goal_car += 2 * np.pi
        self.angle_to_goal = angle_goal_car - self.theta
        if self.angle_to_goal > np.pi:
            self.angle_to_goal -= 2 * np.pi

        # update lidar reading
        self.dist_theta = get_laser_ref(obstacles, self.fov, self.n_reflections, self.max_laser_distance, self.get_robot_pos())

        # save sprime
        self.sp = (deepcopy(self.dist_theta), self.dist_to_goal, self.angle_to_goal * 180 / np.pi, self.get_robot_pos())

    # call update before calling reward
    def reward(self):
        # calculates the cummulative reward of taking an action in s_old and transitioning
        # to a new state by taking an action a
        # returns:
            # reward, game_flag
        
        lidar_old = self.s[0]
        lidar_new = self.sp[0]

        gdist_old = self.s[1]
        gdist_new = self.sp[1]
        
        theta_old = self.s[2]
        theta_new = self.sp[2]

        reward = 0
        end_game = False

########################################################################

        if gdist_new <= 2:
            reward = 10
            end_game = True
            self.goal_reached = True
            print("GOAL")
            return (reward, end_game)
    
        hit_points = np.where(np.array(lidar_new) <= 2)[0]
        if hit_points.size >= 1:
            reward = -10
            end_game = True
            return (reward, end_game)              # return a large negative value and end game
            
        end_game = False

        if theta_old > 0 and self.a[1] > 0 or theta_old < 0 and self.a[1] < 0:
            reward += 0.01 * np.abs(self.a[1]) # 5
        elif theta_old > 0 and self.a[1] < 0 or theta_old < 0 and self.a[1] > 0:
            reward -= 0.03 * np.abs(self.a[1]) # 15

        # 3. if we move towards the goal give reward based on scaling
        # scaling_factor = 0.25 #1 / np.sqrt((100) ** 2 + (50) ** 2)        # WARNING! HARDCODED field dimensions (100, 50)
        # delta_goal_dist = scaling_factor / self.dist_to_goal    # TODO: change a scaling factor potentially
        # reward += delta_goal_dist
 
        diff_arr =  np.array(lidar_new) - np.array(lidar_old)
        diff_arr, wts = get_arr_wts(np.array(diff_arr))        # get the distances in the 180 degree fov and corresponding weights
        
        scaling_factor = 0.1
        cummulative_reward = np.sum(scaling_factor * diff_arr * wts)
        reward += cummulative_reward

########################################################################

        return (reward, end_game)

    def state_numpy(self):
        r = np.empty((len(self.dist_theta) + 5))
        r[:len(self.dist_theta)] = self.dist_theta / float(self.max_laser_distance)
        r[len(self.dist_theta)] = self.s[1]
        r[len(self.dist_theta) + 1] = self.s[2]
        r[len(self.dist_theta) + 2:] = np.array(self.get_robot_pos())
        return r
        
def clamp(value, minimum, maximum):
    if value > maximum:
        return maximum
    elif value < minimum:
        return minimum
    return value

def get_arr_wts(a):
    # remove last value
    a = a[:-1]

    # get num lasers in each quadrant
    n_quad_1 = np.ceil(90 / (360.0 / a.size)).astype(np.int16)
    n_quad_2 = np.floor(90 / (360.0 / a.size)).astype(np.int16)

    # extract and form an array
    arr1 = np.flip(a[:n_quad_1])
    arr2 = np.flip(a[-n_quad_2:])
    arr = np.concatenate((arr1, arr2))
    norm_arr = (np.arange(-n_quad_1 + 1, n_quad_2 + 1))

    mu    = np.mean(norm_arr)
    sigma = np.std(norm_arr)

    wts = []
    for i in range(norm_arr.size):
        wts.append(np.round(0.4 * np.exp(-(0.5) * ((norm_arr[i] - mu) / (sigma)) ** 2), 2))

    return(arr, wts)


def get_laser_ref(segments, fov=np.pi, n_reflections=180, max_dist=100, xytheta_robot=np.array([0.0, 0.0])):
    """
    :param segments: start and end points of all segments as ((x1,y1,x1',y1'), (x2,y2,x2',y2'), (x3,y3,x3',y3'), (...))
           fov: sight of the robot - typically pi or 4/3*pi
           n_reflections: resolution=fov/n_reflections
           max_dist: max distance the robot can see. If no obstacle, laser end point = max_dist
           xy_robot: robot's position in the global coordinate system
    :return: 1xn_reflections array indicating the laser end point
    """
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


def get_intersection(a1, a2, b1, b2) :
    """
    :param a1: (x1,y1) line segment 1 - starting position
    :param a2: (x1',y1') line segment 1 - ending position
    :param b1: (x2,y2) line segment 2 - starting position
    :param b2: (x2',y2') line segment 2 - ending position
    :return: point of intersection, if intersect; None, if do not intersect
    #adopted from https://github.com/LinguList/TreBor/blob/master/polygon.py
    """
    def perp(a) :
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot( dap, dp )

    if denom.astype(float) == 0:
        denom = denom + 1
    intersct = np.array((num/denom.astype(float))*db + b1) ## TODO: check divide by zero!

    delta = 1e-3
    condx_a = min(a1[0], a2[0])-delta <= intersct[0] and max(a1[0], a2[0])+delta >= intersct[0] #within line segment a1_x-a2_x
    condx_b = min(b1[0], b2[0])-delta <= intersct[0] and max(b1[0], b2[0])+delta >= intersct[0] #within line segment b1_x-b2_x
    condy_a = min(a1[1], a2[1])-delta <= intersct[1] and max(a1[1], a2[1])+delta >= intersct[1] #within line segment a1_y-b1_y
    condy_b = min(b1[1], b2[1])-delta <= intersct[1] and max(b1[1], b2[1])+delta >= intersct[1] #within line segment a2_y-b2_y
    if not (condx_a and condy_a and condx_b and condy_b):
        intersct = None #line segments do not intercept i.e. interception is away from from the line segments

    return intersct
