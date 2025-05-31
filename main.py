from stable_baselines3 import PPO
from mapper_env import MapperEnv

if __name__ == "__main__":
    env = MapperEnv()
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000)
