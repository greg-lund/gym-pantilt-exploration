from gym.envs.registration import register

register(
        id='pantilt-exploration-env-v0',
        entry_point='gym_pantilt_exploration.envs:PanTiltEnv',
        kwargs={},
)
