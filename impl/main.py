import numpy as np
from impl.aviary.CustomAviary import CustomAviary
from utils.utils import *
import time, signal,  sys

def run():
    # Define env and forest provider
    forestProvider = ForestProvider(fPoissonGrid=True, fDebug=True)
    env = CustomAviary(
        gui=True, 
        forestProvider=forestProvider
        )

    # Define SIGINT handler
    def sigint_handler(sig, frame):
        # Close and cleanup
        if f_stream:
            close_plt()
        env.info()
        env.close()

        # Exit the program
        print("Shutting down gracefully")
        sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, sigint_handler)

    # Define streaming flag
    f_stream = False
    if f_stream:
        init_plt()
    
    # Define action
    hover_action = 0.0
    
    fYaw = False
    if fYaw:
        yaw_rate = .5
        hover_action += yaw_rate

    final_action = np.array([-hover_action, hover_action, -hover_action, hover_action])

    # Define loop parameters
    sleep_time = 0.02
    logging_interval = 10
    logging_num = 5 # times per `logging_interval`
    logging_freq = logging_interval / logging_num
    count = 1
    while True:
        # Advance
        env.step(final_action)
        
        time.sleep(sleep_time)

        if f_stream:
            rgb_stream(env._getDroneImages(0)[0])

        elapsed_time = count * sleep_time
        if (elapsed_time % logging_freq == 0):
            print(f"Seconds elapsed: {elapsed_time}")
            print(f"Current robot state: {env._computeObs()}")
            # print(f"Current rgb feed:\n {rgb}")


if __name__ == "__main__":
    run()