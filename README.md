# LSTM based Dynamic Obstacle Avoidance for reaching Goal

This was the final project for CS 7649 Robot Intelligence: Planning
We exploit a reinforcement learning technique (simple policy gradient) to train a vehicle with Ackermann steering to reach a goal in an environment containing static or dynamic obstacles.
Future work using an Actor-Critic method may be done to improve model performance.

Check RIP_Report.pdf for workshop-style report.
Demonstration: https://youtu.be/Tj5GInNJTI0

## Installation
conda create --name env_name --file Requirements.txt
  
conda activate env_name

### Config
We have enviroments under the config folder. We have a few pre-made ones and have also provided the code to create environments with dynamic obstacles.

### Saved weights
We have provided you with two weights (static_starting.pt and dynamic_obs.pt)

## Train the model
Please refer to the train method in `pg-lstm.py` to change the enviroment. Default values of the iteration is 2000 and rollout is 200.
Run `python3 pg-lstm.py` to train.

## Visualization
There are 5 methods in `animation_replay.py` that can visualize the results shown in the report.
Please uncomment the function you would like to visualize in the main method at the bottom.
no_obs() - No obstacles
static_fail() - Static obstacle (failure)
static_success() - Static obstacle (success)
dynamic_fail() - Dynamic obstacle (failure)
dynamic_success() - Dynamic obstacle (success)
Run `python3 animation_replay.py` to run the visualization. The rewards will be printed to stdout.


### Keyboard control of the agent
Please run `python3 animation_control.py` for being able to control the agent using your keyboard (left & right arrow keys)
