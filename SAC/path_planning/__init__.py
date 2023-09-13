from gym.envs.registration import register
register(
    id='Path_planning-v0',
    entry_point='path_planning.envs:Kukaenv'
)