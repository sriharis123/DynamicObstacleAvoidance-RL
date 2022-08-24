"""
This simulator can be used to generate laser reflections in dynamic environments.
Ransalu Senanayake
06-Aug-2018

Step 1: Specify the output folder and file name
Step 2: Specify the environemnt (use an existing or create your own)
Step 3: Specify the lidar parameters such as distance, angle, etc,
Step 4: Draw the robot's path by clicking on various locations on the gui (close to exit) or hard code the pose/s
Output: .csv or carmen file and images

Note: Output file type .csv: column1=time, column2=longitude, column3=latitude, column4=occupied/free
"""

from matplotlib import animation
import matplotlib.pylab as pl
import numpy as np
import keyboard
from car import Car
from animation_util import load_goal_config, load_obstacles_config, connect_segments, load_robot_config

# file configuration
# env = 'toy1' # name of the yaml file inside the config folder
env = 'static_obstacle'

# robot LIDAR configuration
n_reflections = 18 # number of lidar beams in the 2D plane
fov = 360 *np.pi/180 # lidar field of view in degrees
max_laser_distance = 10 # maximum lidar distance in meters
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
phi_init = 0
dt           = 0.1  # seconds

goal_loc     = [goal_loc[0], goal_loc[1]]
max_velocity = 10
min_velocity = -6

car = Car(x_init, y_init, L, theta_init, phi_init, velocity_init, dt, goal_loc, max_velocity, min_velocity, 
        n_reflections=n_reflections, fov=fov, max_laser_distance=max_laser_distance, unoccupied_points_per_meter=unoccupied_points_per_meter)

# set up initial lidar reading
# update obstacles
all_obstacle_segments = []
for obs_i in all_obstacles:
    all_obstacle_segments += obs_i.update(0)

# get the new lidar points
car.update(0, 0, all_obstacle_segments)

fig = pl.figure()
ax = fig.add_subplot(111)
    
end_game = False

def game(i):

    ###
    # Car is in the current state
    v_dot = 0
    phi_dot = 0

    # provide an action to the car in the current state
    if keyboard.is_pressed('right'):
        phi_dot = -25 * np.pi / 180
    
    if keyboard.is_pressed('left'):
        phi_dot = 25 * np.pi / 180

    if keyboard.is_pressed('up'):
        v_dot = 7
    
    if keyboard.is_pressed('down'):
        v_dot = -7

    #update obstacles
    all_obstacle_segments = []
    for obs_i in all_obstacles:
        all_obstacle_segments += obs_i.update(car.dt)
    
    # get the new lidar points
    car.update(v_dot, phi_dot, all_obstacle_segments)
    print(car.reward()[0])
   
    robot_pos = car.get_robot_pos()
    dist_theta = car.dist_theta
    

    ##################################################################################################################3
    # populate with zeros in-between the robot and laser hit. the density parameter is 'unoccupied_points_per_meter'
    # laser_data_xyout_filled = get_filled_txy(dist_theta, robot_pos, car.fov, car.n_reflections, car.max_laser_distance,
    #                                             car.unoccupied_points_per_meter)

    # (x,y) of laser reflections
    angles = np.linspace(robot_pos[2], robot_pos[2] + car.fov, car.n_reflections)
    laser_data_xy = np.vstack([dist_theta*np.cos(angles), dist_theta*np.sin(angles)]).T + robot_pos[:2]
    # get the environment for plotting purposes
    connected_components = connect_segments(all_obstacle_segments)

    ax.clear()
    ax.scatter(connected_components[:,0], connected_components[:,1], marker='.', c='y', edgecolor=['none'], alpha=0.2) #obstacles
    for i in range(car.n_reflections): #laser beams
        laser_color = 'blue'
        if i == 0 or i == 359:
            laser_color = 'red'
        ax.plot(np.asarray([robot_pos[0], laser_data_xy[i, 0]]), np.asarray([robot_pos[1], laser_data_xy[i, 1]]), c=laser_color, zorder=1, alpha=0.2)
        if dist_theta[i] < car.max_laser_distance:
            ax.scatter(laser_data_xy[i,0], laser_data_xy[i,1], marker='o', c='r', zorder=2, edgecolor=['none']) #laser end points
    ax.scatter([0], [0], marker='*', c='k', s=20, alpha=1.0, zorder=3, edgecolor='k')  # global origin
    ax.scatter(robot_pos[0], robot_pos[1], marker=(3, 0, robot_pos[2]/np.pi*180 + 30), c='k', s=300, alpha=1.0, zorder=3, edgecolor='k')#robot's position
    # ax.scatter(robot_pos[0], robot_pos[1], marker=(4, 0, robot_pos[2]/np.pi*180), c='k', s=300, alpha=1.0, zorder=3, edgecolor='k')#robot's position
    ax.scatter(car.goal_loc[0], car.goal_loc[1], marker='*', s=300, c='g')
    #ab = AnnotationBbox(getImage(robot_pos[2] * 180 / np.pi), (robot_pos[0], robot_pos[1]), frameon=False)
    #ax.add_artist(ab)

    #ax.plot(robot_poses[:,0], robot_poses[:,1], 'k--')
    ax.set_xlim([area[0], area[1]])
    ax.set_ylim([area[2], area[3]])
    pl.tight_layout()
    ##################################################################################################################3

ani = animation.FuncAnimation(fig, game, 10000, repeat=not end_game)       # will fun the game function every 10 ms and plot the output as well

pl.show()