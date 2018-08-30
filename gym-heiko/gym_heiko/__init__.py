from gym.envs.registration import register

# add more environments if needed

register(
    id='heikopendulum-v0',
    entry_point='gym_heiko.envs:HeikoPendulumEnv',
    max_episode_steps=200,
)
register(
    id='heikopendulumfmu-v0',
    entry_point='gym_heiko.envs:HeikoPendulumFMUEnv',
    max_episode_steps=200,
)
