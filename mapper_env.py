import pathlib
import gym
import numpy as np
import sys, types, pathlib
from hydra import compose, initialize

_repo = pathlib.Path(__file__).resolve().parent / "Mapperatorinator"
if _repo.exists():
    sys.path.insert(0, str(_repo))
    osuT5_root = _repo / "osuT5"
    if osuT5_root.exists():
        sys.path.insert(0, str(osuT5_root))
        if "osuT5" not in sys.modules:
            pkg = types.ModuleType("osuT5")
            pkg.__path__ = [str(osuT5_root)]
            sys.modules["osuT5"] = pkg

from reward_function import evaluate_osu_file
from Mapperatorinator.inference import main


class MapperEnv(gym.Env):
    def __init__(self, audio="audios/audio.mp3", osu_folder="output_maps"):
        super().__init__()

        # 參數：7個 continuous + 1個 binary（super_timing）
        self.action_space = gym.spaces.Box(
            low=np.array([2.0, 3.0, 5.0, 0.8, 0.6, 1.0, 0.0]),  # 最後是 super_timing
            high=np.array([6.0, 10.0, 10.0, 2.0, 1.4, 10.0, 1.0]),
            dtype=np.float32,
        )

        # 觀察空間這裡先簡化為空向量（黑盒）
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        with initialize(config_path="Mapperatorinator/configs", version_base="1.1"):
            self.cfg = compose(
                config_name="inference_v30",
                overrides=[
                    f"audio_path={audio}",
                    f"output_path={osu_folder}",
                    "export_osz=True",
                ],
            )

    def reset(self):
        # 可以加一些隨機初始化邏輯
        return np.zeros((1,), dtype=np.float32)

    def step(self, action):
        # 解碼參數
        self._decode_action(action)

        # 跑 Mapperatorinator 並產生 .osu
        osu_path = self._generate_map()  # removed self.cfg parameter

        # 計算 reward
        reward = evaluate_osu_file(osu_path)

        # 單步問題，done 一律為 True
        return np.zeros((1,), dtype=np.float32), reward, True, {}

    def _decode_action(self, action):
        # 將連續空間值轉成實際 config 值 using self.cfg instead of self.conf
        self.cfg.circle_size = float(action[0])
        self.cfg.overall_difficulty = float(action[1])
        self.cfg.approach_rate = float(action[2])
        self.cfg.slider_multiplier = float(action[3])
        self.cfg.temperature = float(action[4])
        self.cfg.cfg_scale = float(action[5])
        self.cfg.super_timing = bool(round(action[6]))

    def _generate_map(self):
        # Use self.cfg instead of self.conf when calling main()
        _, result_path, osz_path = main(self.cfg)
        if osz_path is not None:
            result_path = osz_path
        return result_path


# Added RL training function based on test.py and Mapperatorinator content
def train_rl(num_episodes=10):
    # Initialize environment: update audio path to audios/audio.mp3
    env = MapperEnv(audio="audios/audio.mp3")
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            total_reward += reward
        print(f"Episode {episode + 1}: Total Reward = {total_reward}")


if __name__ == "__main__":
    train_rl()
