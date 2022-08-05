# Adroid-RL. Based on [gym_pybullet_drones](https://github.com/utiasDSL/gym-pybullet-drones)

## Separation rationale
Initially this project was implemented as a package within [gym_pybullet_drones](https://github.com/utiasDSL/gym-pybullet-drones) fork.
But everything has changed after a [Massive Purge](https://github.com/hidal00p/adroid-rl/commit/4477392deea42d55b5a0100d870de7dcc4d41a8d). This was done to favour modularity, and compactness of the project.
Adroid-RL, however still heavily relates on its parent. And in order to build it successfully, a proper configuration for [gym_pybullet_drones](https://github.com/utiasDSL/gym-pybullet-drones) has to be ensured first.

## Checkpoints
- Checkpoint 1; commit: 7e9429c31e3a96dbd9de311df26a9867f1062c5e