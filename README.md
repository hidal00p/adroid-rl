# Adroid-RL. Based on [gym_pybullet_drones](https://github.com/utiasDSL/gym-pybullet-drones)

## Separation rationale
Initially this project was implemented as a package within [gym_pybullet_drones](https://github.com/utiasDSL/gym-pybullet-drones) fork.
But everything has changed after a [Massive Purge](https://github.com/hidal00p/adroid-rl/commit/4477392deea42d55b5a0100d870de7dcc4d41a8d). This was done to favour modularity, and compactness of the project.
Adroid-RL, however still heavily relates on its parent. And in order to build it successfully, a proper configuration for [gym_pybullet_drones](https://github.com/utiasDSL/gym-pybullet-drones) has to be ensured first.

## Checkpoints
Once checkpoint_n+1 appears, checkpoint_n get erased, unless decided otherwise.

- ~~Checkpoint 1; commit: 7e9429c31e3a96dbd9de311df26a9867f1062c5e~~
- Checkpoint 2; commit: 90b4e17db81f4666cd34e137db93d0f5c50332c4

### Rationale
It is a spin-off of Checkpoint 2 (ch-2) branch. 
The idea is to encode a chance of allowing the env to remain as is, with only a bait updating its position.
Hopefully this would include a greater sense of continuity into the environment and the mission.

If it passes the performance test, and sets a new bench, I will add the configurable param for this setting.