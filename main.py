import numpy as np
import time, signal, sys

import aviary.utils as au
from utils import closePlt, initPlt, rgbStream

def main():
    env = au.getEnv(fGui=True)
    
    env._initReferencePath()
    env.agentInfo()

    # Define SIGINT handler
    def sigintHandler(sig, frame):
        env.obstacleInfo()
        env.close()
        
        # Close and cleanup
        if f_stream:
            closePlt()

        # Exit the program
        print("Shutting down gracefully")
        sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, sigintHandler)

    # Define streaming flag
    f_stream = True
    if f_stream:
        initPlt()
    
    # Define action
    hover_action = 0.0
    
    fYaw = True
    if fYaw:
        yaw_rate = .1
        hover_action += yaw_rate

    final_action = np.array([-hover_action, hover_action, -hover_action, hover_action])

    # Define loop parameters
    sleep_time = 0.02
    
    logging_interval = 1
    logging_num = 4 # times per `logging_interval`
    logging_freq = logging_interval / logging_num

    streaming_interval = 1
    streaming_num = 2
    streaming_freq = streaming_interval / streaming_num

    count = 1
    elapsed_time = 0
    while True:
        # Advance
        env.step(final_action)
        
        if (f_stream and (elapsed_time % streaming_freq == 0)):
            rgbStream(env._getDroneImages(0)[0])

        if (elapsed_time % logging_freq == 0):
            agenOr = env.extractAgentOrientationVector()
            baitOr = env.computeBaitCompass()
            print(f"Seconds elapsed: {elapsed_time}")
            print(f"Robot spatial data: {agenOr}")
            print(f"Bait compass spatial data: {baitOr}")
            print(f"Directional correlation: {agenOr.correlateTo(baitOr)}")
            # print(f"Current robot state: {env._computeObs()}")
            # env.baitInfo()
            # env.agentInfo()
            
        elapsed_time = count * sleep_time
        count += 1
        time.sleep(sleep_time)

if __name__ == "__main__":
    main()
