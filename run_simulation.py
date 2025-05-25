from env.microgrid_env import MicrogridEnv

env = MicrogridEnv()
obs = env.reset()
done = False

print("Initial observation:", obs)

while not done:
    action = env.action_space.sample()  # Random action for now
    obs, reward, done, info = env.step(action)
    print(f"Step: {env.current_step}, Action: {action}, Done: {done}")

env.render()  # Open visual layout of the network
