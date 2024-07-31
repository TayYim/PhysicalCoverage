import os
import sys
import gym
import time
import argparse
import datetime
import highway_env_v2
import numpy as np
import matplotlib.pyplot as plt

from copy import copy
from coverage import Coverage
from math import pi, atan2, degrees

# Hot fix to get general accepted
from pathlib import Path
current_file = Path(__file__)
path = str(current_file.absolute())
base_directory = str(path[:path.rfind("/environments/highway")])
sys.path.append(base_directory)

from utils.reachset import ReachableSet
from utils.environment_configurations import RRSConfig
from utils.environment_configurations import HighwayKinematics
from utils.fingerprint import compute_grid, get_planning_type, get_points

from highway_config import HighwayEnvironmentConfig

from controllers.tracker import Tracker
from controllers.car_controller import EgoController

# Get the different configurations
HK = HighwayKinematics()
RRS = RRSConfig(beam_count=31)

# Variables - Used for timing
total_lines     = RRS.beam_count
steering_angle  = HK.steering_angle
max_distance    = HK.max_velocity

# Declare the obstacle size (1 - car; 0.5 - motorbike)
obstacle_size = 1

# Suppress exponential notation
np.set_printoptions(suppress=True)

# Create the controllers
hw_config = HighwayEnvironmentConfig(environment_vehicles=5, duration=10, crash_ends_test=False)
car_controller = EgoController(debug=False)
tracker = Tracker(distance_threshold=5, time_threshold=0.5, debug=False)
reach = ReachableSet(obstacle_size=obstacle_size)

# Create the environment
env = gym.make("highway-v0")
env.config = hw_config.env_configuration
env.reset()

# Default action is IDLE
action = car_controller.default_action()

# Get the roadway - used when calculating the edge of the road
lanes = env.road.network.graph['0']['1']
lane_width = np.array([0, lanes[0].width/2.0])

# Main loop
done = False

# Init timing variables
start_time = datetime.datetime.now()
simulated_time_counter = 0
simulated_time_period = 1.0 / hw_config.policy_freq
first = True
total_physical_accidents = 0
while not done:

    # Increment time
    simulated_time_counter += 1

    # Step the environment
    obs, reward, done, info = env.step(action)
    obs = np.round(obs, 4)

    # Print the observation and crash data
    print("Environment:")
    print("|--Crash: \t\t" + str(info["crashed"]))
    print("|--Collided: \t\t" + str(info["collided"]))
    print("|--Speed: \t\t" + str(np.round(info["speed"], 4)))
    print("|--Observation: \n" + str(obs))
    print("")

    # Get the next action based on the current observation
    action = car_controller.drive(obs)

    # Track objects
    tracker.track(obs)
    tracked_objects = tracker.get_observations()

    # Track the time for this operation
    op_start_time = datetime.datetime.now()

    # Convert the lane positions to be relative to the ego_vehicle
    # 计算车道的相对起点终点位置
    ego_position = env.controlled_vehicles[0].position
    upperlane = [lanes[0].start-lane_width, lanes[0].end-lane_width] - ego_position
    lowerlane = [lanes[-1].start+lane_width, lanes[-1].end+lane_width] - ego_position
    lane_positions = [upperlane, lowerlane]

    # Get the reach set simulation
    polygons    = reach.compute_environment(tracked_objects, lane_positions)
    r_set       = reach.estimate_raw_reachset(total_lines=total_lines, 
                                              steering_angle=steering_angle,
                                              max_distance=max_distance)
    final_r_set = reach.estimate_true_reachset(polygons, r_set)
    r_vector    = reach.vectorize_reachset(final_r_set, accuracy=0.001)

    # =================================================================================================
    # Go our method here

    edge_length = 60
    cell_size = 4

    l = get_points(edge_length, edge_length, tracked_objects, lane_positions)
    grid = compute_grid(edge_length, cell_size, l)

    print(grid)

    # grid to binary numpy 1D array
    grid_flatten = grid.flatten()

    # grid_flatten to binary string
    grid_str = ''.join(map(str, grid_flatten))
    print(grid_str)

    # grid_str to decimal
    grid_decimal = int(grid_str, 2)
    print(grid_decimal)

    # > planning
    planning = get_planning_type(car_controller)

    # Show fig
    plt.figure(1)
    plt.clf()
    plt.title('Environment')

    # Invert the y axis for easier viewing
    plt.gca().invert_yaxis()

    # Set the size of the graph
    plt.xlim([-edge_length//2, edge_length//2])
    plt.ylim([-edge_length//2, edge_length//2])

    # Invert the y axis as negative is up and show ticks
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)

    for i in range(edge_length // cell_size + 1):
        ax.axhline(i * cell_size - edge_length//2, color='grey', linewidth=0.5)
        ax.axvline(i * cell_size - edge_length//2, color='grey', linewidth=0.5)

    l_clean = [(x, y) for (x, y) in l if -edge_length//2 <= x <= edge_length//2 and -edge_length//2 <= y <= edge_length//2]
    x_coords, y_coords = zip(*l_clean)
    ax.scatter(y_coords, x_coords, color='red') # reverse for better view

    # 设置网格
    major_ticks = np.arange(-edge_length//2, edge_length//2+1, cell_size)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # draw ego in (0, 0) using green
    ax.scatter(0, 0, color='green')
    # draw arrow for planning
    if planning == 1:
        ax.arrow(0, 0, 5, 2, head_width=0.5, head_length=0.5, fc='black', ec='black')
    elif planning == -1:
        ax.arrow(0, 0, 5, -2, head_width=0.5, head_length=0.5, fc='black', ec='black')
    else:
        ax.arrow(0, 0, 5, 0, head_width=0.5, head_length=0.5, fc='black', ec='black')

    # plot the graph
    # >= 0.3 to see the points
    plt.pause(0.3)


    # =================================================================================================

    # Track the time for this operation
    current_time = datetime.datetime.now()
    operation_time = (current_time - op_start_time).total_seconds()
    elapsed_time = (current_time - start_time).total_seconds()

    print("")
    print("Vector: " + str(r_vector))
    print("---------------------------------------")

    env.render()

    simulated_time = np.round(simulated_time_period * simulated_time_counter, 4)

    # If it crashed determine under which conditions it crashed
    if info["collided"]:
        total_physical_accidents += 1
        try:
            # Get the velocity of the two vehicles (we want the velocities just before we crashed)
            ego_vx, ego_vy = info["kinematic_history"]["velocity"][1]
            veh_vx, veh_vy = info["incident_vehicle_kinematic_history"]["velocity"][1]

            # Get magnitude of both velocity vectors
            ego_mag = np.linalg.norm([ego_vx, ego_vy])
            veh_mag = np.linalg.norm([veh_vx, veh_vy])

            # Get the angle of incidence
            angle_of_incidence = degrees(atan2(veh_vy, veh_vx) - atan2(ego_vy, ego_vx))

            # Round all values to 4 decimal places
            ego_mag = np.round(ego_mag, 4)
            veh_mag = np.round(veh_mag, 4)
            angle_of_incidence = np.round(angle_of_incidence, 4)
        except ValueError:
            ego_mag = 0
            veh_mag = 0 
            angle_of_incidence = 0

        print("Ego velocity magnitude: {}".format(ego_mag))
        print("Incident vehicle velocity magnitude: {}".format(veh_mag))
        print("Angle of incident: {}".format(angle_of_incidence))
        print("")
        print("---------------------------------------")


print("-----------------------------")


env.close()
