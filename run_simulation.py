from env.microgrid_env import MicrogridEnv

env = MicrogridEnv()
obs = env.reset()

for _ in range(10):
    obs, reward, done, info = env.step(env.action_space.sample())
    print(f"Step {_+1}, Done: {done}")

env.render()
