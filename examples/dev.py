import os
import time
from datetime import datetime
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import ray
from ray import tune
from ray.rllib.agents import ppo
from ray.rllib.utils.test_utils import check_learning_achieved

from utils import *
from gym_pybullet_drones.envs.Aviary import DroneModel, Physics, Aviary
from gym_pybullet_drones.envs.Logger import Logger
from gym_pybullet_drones.envs.Control import ControlType, Control
from gym_pybullet_drones.envs.RLFunctions import Problem 

DRONE = DroneModel.CF2X
NUM_DRONES = 3
GUI = False
PHYSICS = Physics.PYB
RECORD_VIDEO = False
SIMULATION_FREQ_HZ = 240
CONTROL_FREQ_HZ = 48
DURATION_SEC = 10

if __name__ == "__main__":

    if False:

        #### Initialize the simulation #####################################################################
        H = .1; H_STEP = .05; R = .3; INIT_XYZS = np.array([ [R*np.cos((i/6)*2*np.pi+np.pi/2), R*np.sin((i/6)*2*np.pi+np.pi/2)-R, H+i*H_STEP] for i in range(NUM_DRONES) ])
        env = Aviary(drone_model=DRONE, num_drones=NUM_DRONES, initial_xyzs=INIT_XYZS, physics=PHYSICS, visibility_radius=10, \
                        normalized_spaces=False, freq=SIMULATION_FREQ_HZ, gui=GUI, obstacles=True, record=RECORD_VIDEO, problem=Problem.MA_FLOCK); env.reset()

        #### Initialize a circular trajectory ##############################################################
        PERIOD = 10; NUM_WP = CONTROL_FREQ_HZ*PERIOD; TARGET_POS = np.zeros((NUM_WP,3))
        for i in range(NUM_WP): TARGET_POS[i,:] = R*np.cos((i/NUM_WP)*(2*np.pi)+np.pi/2)+INIT_XYZS[0,0], R*np.sin((i/NUM_WP)*(2*np.pi)+np.pi/2)-R+INIT_XYZS[0,1], INIT_XYZS[0,2]  
        wp_counters = np.array([ int((i*NUM_WP/6)%NUM_WP) for i in range(NUM_DRONES) ])
        
        #### Initialize the logger #########################################################################
        logger = Logger(duration_sec=DURATION_SEC, simulation_freq_hz=SIMULATION_FREQ_HZ, num_drones=NUM_DRONES)

        #### Initialize the controllers ####################################################################    
        ctrl = [Control(env, control_type=ControlType.PID) for i in range(NUM_DRONES)]

        #### Run the simulation ############################################################################
        CTRL_EVERY_N_STEPS= int(np.floor(env.SIM_FREQ/CONTROL_FREQ_HZ))
        action = { str(i): np.array([0,0,0,0]) for i in range(NUM_DRONES) } if NUM_DRONES>1 else np.array([0,0,0,0])
        START = time.time(); temp_action = {}
        for i in range(DURATION_SEC*env.SIM_FREQ):

            #### Step the simulation ###########################################################################
            obs, reward, done, info = env.step(action)

    ##############################
    ##############################
    ##############################
            print("Obs", obs)
            print("Norm Obs", {str(i): env.RL_FUNCTIONS.clipAndNormalizeState(obs[str(i)]["state"], env.step_counter) for i in range(NUM_DRONES)})
            print("Reward", reward)
            print("Done", done)
            print()
    ##############################
    ##############################
    ##############################

            #### Transform 1-drone obs into the Dict format of multiple drones to simplify the code ############
            if NUM_DRONES==1: obs = {"0": {"state": obs}}

            #### Compute control at the desired frequency @@@@@#################################################       
            if i%CTRL_EVERY_N_STEPS==0:

                #### Compute control for the current waypoint ######################################################
                for j in range(NUM_DRONES): 
                    temp_action[str(j)], _, _ = ctrl[j].computeControlFromState(control_timestep=CTRL_EVERY_N_STEPS*env.TIMESTEP, \
                                                                                    state=obs[str(j)]["state"], \
                                                                                    target_pos=np.hstack([TARGET_POS[wp_counters[j],0:2], H+j*H_STEP]))
                    
                    #### Transform multi-drone actions into the Box format of a single drone to simplify the code ######
                    action = temp_action if NUM_DRONES>1 else temp_action["0"]

                #### Go to the next waypoint and loop ##############################################################
                for j in range(NUM_DRONES): wp_counters[j] = wp_counters[j] + 1 if wp_counters[j]<(NUM_WP-1) else 0

            #### Log the simulation ############################################################################
            for j in range(NUM_DRONES): logger.log(drone=j, timestamp=i/env.SIM_FREQ, state= obs[str(j)]["state"], control=np.hstack([ TARGET_POS[wp_counters[j],0:2], H+j*H_STEP, np.zeros(9) ]))   
            
            #### Printout ######################################################################################
            env.render()

            #### Sync the simulation ###########################################################################
            if GUI: sync(i, START, env.TIMESTEP)
       
        #### Close the environment #########################################################################
        env.close()

        #### Save the simulation results ###################################################################
        logger.save()

        #### Plot the simulation results ###################################################################
        logger.plot()

    else:

        #### WIP ###########################################################################################

        config = {
            "env": Aviary,
            "num_workers": 2,
            "env_config": {
                "drone_model": DRONE,
                "num_drones": NUM_DRONES,
                "visibility_radius": np.inf,
                "initial_xyzs": None,
                "initial_rpy": None,
                "physics": PHYSICS,
                "normalized_spaces": True,
                "freq": SIMULATION_FREQ_HZ,
                "gui": False,
                "obstacles": False,
                "record": False,
                "problem": Problem.MA_FLOCK,
            },
            # "multiagent": {
            #     "policies": {
            #         "pol1": (None, o_s, a_s, {"agent_id": 0,}),
            #         "pol2": (None, o_s, a_s, {"agent_id": 1,}),
            #     },
            #     "policy_mapping_fn": lambda agent_id: "pol1" if agent_id == 0 else "pol2",
            # },
        }

        stop = {
            "timesteps_total": 10000,
        }

        ray.shutdown()
        ray.init(ignore_reinit_error=True)
        print("Dashboard URL: http://{}".format(ray.get_webui_url()))

        results = tune.run("PPO", stop=stop, config=config, verbose=True)

        check_learning_achieved(results, 1.0)

        ray.shutdown()

