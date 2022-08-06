# Adroid-RL. Based on [gym_pybullet_drones](https://github.com/utiasDSL/gym-pybullet-drones)

## Separation rationale
Initially this project was implemented as a package within [gym_pybullet_drones](https://github.com/utiasDSL/gym-pybullet-drones) fork.
But everything has changed after a [Massive Purge](https://github.com/hidal00p/adroid-rl/commit/4477392deea42d55b5a0100d870de7dcc4d41a8d). This was done to favour modularity, and compactness of the project.
Adroid-RL, however still heavily relates on its parent. And in order to build it successfully, a proper configuration for [gym_pybullet_drones](https://github.com/utiasDSL/gym-pybullet-drones) has to be ensured first.

## Branch strcture
This project uses branches in a somewhat dual way.
1. In a standard git understanding of a branch
2. As a checkpoint list for which a particular hypothesis succeeded. Thus each branch marked as ch-{N} contains a checkpoint with potentially some documentation related to it.

## Signatures or traits
It is a naming convention for hypothesis runs, which has a self-documenting character.
There is no particular convention apart from the following one:

- Separate logically different signatures with `-`
- Unite multiple words in logically similar signatures with `_`
- Try to capture the following categories in signatures:
    - Algorithm eg `sac`, `ppo`, `ddpg`
    - Reward and penalty policy eg `severe_scold`, `relaxed_scold`, `agressive_penalty`, `generous_rew` etc
    - Episodicity eg `fin_horizon`, `inf_hor`
    - Geometric boundary specifics `xyz_boundary`, `z_limit`, `relaxed_z_lim`
    - Observation struct `messy_obs`, `kinem_obs`, `imu_measur`, `rgb_depth_vec` etc
    - Goal description `seek_and_find`, `obs_av`, `nav`, `move_to`


**Example signature:**

`sac-xy_z_diff-diff_xyz_obs-strict_death-simple_find`

### Extended signatures
Extended signatures are tailed with arch specifics, or some of your test env sepcifics like total time steps etc.

**Extended Example signature:**

`(ppo-xy-diff_obs-simple_find)-relu-500000-120deg-121-384-256`

## Checkpoints
Once checkpoint_n+1 appears, checkpoint_n get erased, unless decided otherwise.

- ~~Checkpoint 1; commit: 7e9429c31e3a96dbd9de311df26a9867f1062c5e~~
- Checkpoint 2; ch-2:90b4e17db81f4666cd34e137db93d0f5c50332c4