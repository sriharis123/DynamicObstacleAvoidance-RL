"""
Play Animation from rollout
"""

from matplotlib import animation
import numpy as np
import torch
import matplotlib.pylab as pl
from animation_util import connect_segments
import simulator
import importlib
pg = importlib.import_module("pg-lstm")
import torch
from car import Car
from animation_util import load_obstacles_config, connect_segments

import importlib
pg = importlib.import_module("pg-lstm")

class Replay():
    def __init__(self, _fig, _ax, _tau, _obstacles, _area, _car, _env='static_180', _rate=100):
        self.fig = _fig
        self.ax = _ax
        self.tau = _tau
        self.obstacles = _obstacles
        self.area = _area
        self.car = _car
        self.env = _env
        self.rate = _rate
        self.end_game = False

    def animate(self, i):
        if i >= len(self.tau):
            return

        s, a, sp, r = self.tau[i]
        print(r)
        all_obstacle_segments = self.obstacles[i]

        dist_theta, dist_to_goal, angle_to_goal, robot_pos = s
        
        self.end_game = i == len(self.tau) - 1  # TODO: animation end condition not working for some reason

        ##################################################################################################################3
        # populate with zeros in-between the robot and laser hit. the density parameter is 'unoccupied_points_per_meter'
        # laser_data_xyout_filled = get_filled_txy(dist_theta, robot_pos, self.car.fov, self.car.n_reflections, self.car.max_laser_distance,
        #                                             self.car.unoccupied_points_per_meter)

        # (x,y) of laser reflections
        angles = np.linspace(robot_pos[2], robot_pos[2] + self.car.fov, self.car.n_reflections)
        laser_data_xy = np.vstack([dist_theta*np.cos(angles), dist_theta*np.sin(angles)]).T + robot_pos[:2]
        # get the environment for plotting purposes
        connected_components = connect_segments(all_obstacle_segments)

        self.ax.clear()
        self.ax.scatter(connected_components[:,0], connected_components[:,1], marker='.', c='y', edgecolor=['none'], alpha=0.2) #obstacles
        for i in range(self.car.n_reflections): #laser beams
            laser_color = 'blue'
            if i == 0 or i == 359:
                laser_color = 'red'
            self.ax.plot(np.asarray([robot_pos[0], laser_data_xy[i, 0]]), np.asarray([robot_pos[1], laser_data_xy[i, 1]]), c=laser_color, zorder=1, alpha=0.2)
            if dist_theta[i] < self.car.max_laser_distance:
                self.ax.scatter(laser_data_xy[i,0], laser_data_xy[i,1], marker='o', c='r', zorder=2, edgecolor=['none']) #laser end points
        self.ax.scatter([0], [0], marker='*', c='k', s=20, alpha=1.0, zorder=3, edgecolor='k')  # global origin
        self.ax.scatter(robot_pos[0], robot_pos[1], marker=(3, 0, robot_pos[2]/np.pi*180 + 30), c='k', s=300, alpha=1.0, zorder=3, edgecolor='k')#robot's position
        # self.ax.scatter(robot_pos[0], robot_pos[1], marker=(4, 0, robot_pos[2]/np.pi*180), c='k', s=300, alpha=1.0, zorder=3, edgecolor='k')#robot's position
        self.ax.scatter(self.car.goal_loc[0], self.car.goal_loc[1], marker='*', s=300, c='g')
        #ab = AnnotationBbox(getImage(robot_pos[2] * 180 / np.pi), (robot_pos[0], robot_pos[1]), frameon=False)
        #ax.add_artist(ab)

        #ax.plot(robot_poses[:,0], robot_poses[:,1], 'k--')
        self.ax.set_xlim([self.area[0], self.area[1]])
        self.ax.set_ylim([self.area[2], self.area[3]])
        pl.tight_layout()
        ##################################################################################################################3


def run_animation(_tau, _obstacles, _area, _car, _env='static_180', _rate=100):
    fig = pl.figure()
    ax = fig.add_subplot(111)

    a = Replay(fig, ax, _tau, _obstacles, _area, _car, _env, _rate)

    ani = animation.FuncAnimation(fig, a.animate, frames=len(_tau), interval=_rate, repeat=False)       # will fun the game function every 10 ms and plot the output as well
    pl.show()

def no_obs():
    lstm.load_state_dict(torch.load("static_starting.pt"))
    tau, obs, area, car = simulator.rollout(lstm, env='ex_static_no_obs_success', rolloutLimit=1000, isTest=True)
    run_animation(tau,obs,area,car, _env='ex_static_no_obs_success')

def static_fail():
    lstm.load_state_dict(torch.load("static_starting.pt"))
    tau, obs, area, car = simulator.rollout(lstm, env='ex_static_obs_fail', rolloutLimit=1000, isTest=True)
    run_animation(tau,obs,area,car, _env='ex_static_obs_fail')

def static_success():
    lstm.load_state_dict(torch.load("static_starting.pt"))
    tau, obs, area, car = simulator.rollout(lstm, env='ex_static_obs_success', rolloutLimit=1000, isTest=True)
    run_animation(tau,obs,area,car, _env='ex_static_obs_success')

def dynamic_fail():
    lstm.load_state_dict(torch.load("dynamic_obs.pt"))
    tau, obs, area, car = simulator.rollout(lstm, env='ex_dynamic_obstacle_fail', rolloutLimit=1000, isTest=True)
    run_animation(tau,obs,area,car, _env='ex_dynamic_obstacle_fail')

def dynamic_success():
    lstm.load_state_dict(torch.load("dynamic_obs.pt"))
    tau, obs, area, car = simulator.rollout(lstm, env='ex_dynamic_obstacle_success', rolloutLimit=1000, isTest=True)
    run_animation(tau,obs,area,car, _env='ex_dynamic_obstacle_success')

if __name__ == "__main__":

    lstm = pg.LSTM(23, 36, 2)
    
    #no_obs()
    #static_fail()
    static_success()
    #dynamic_fail()
    #dynamic_success()

