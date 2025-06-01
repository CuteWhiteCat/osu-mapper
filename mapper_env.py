import pathlib
import gym
import numpy as np
import sys, types, pathlib
from hydra import compose, initialize
import random

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

        with initialize(config_path="Mapperatorinator/configs", version_base="1.1"):
            self.cfg = compose(
                config_name="inference_v29",
                overrides=[
                    f"audio_path={audio}",
                    f"output_path={osu_folder}",
                    "export_osz=True",
                    "difficulty=6",
                ],
            )

    def step(self, action):
        # 解碼參數
        self._decode_action(action)

        # 跑 Mapperatorinator 並產生 .osu
        osu_path = self._generate_map()

        # 計算 reward
        reward = evaluate_osu_file(osu_path)

        return reward

    def _decode_action(self, action):
        # 將連續空間值轉成實際 config 值 using self.cfg instead of self.conf
        self.cfg.circle_size = float(action[0])
        self.cfg.overall_difficulty = float(action[1])
        self.cfg.approach_rate = float(action[2])
        self.cfg.slider_multiplier = float(action[3])
        self.cfg.temperature = float(action[4])
        self.cfg.cfg_scale = float(action[5])
        self.cfg.super_timing = bool(action[6])
        self.cfg.mapper_id = int(action[7])
        self.cfg.year = int(action[8])
        self.cfg.descriptors = [str(action[9])]
        self.cfg.negative_descriptors = [str(action[10])]
        self.cfg.seed = random.randint(0, 2147483647)

    def _generate_map(self):
        # Use self.cfg instead of self.conf when calling main()
        _, result_path, osz_path = main(self.cfg)
        if osz_path is not None:
            result_path = osz_path
        return result_path

