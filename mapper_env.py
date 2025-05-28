import gym
import numpy as np
import tempfile
import os
from reward_function import evaluate_osu_file  # 你自訂的 reward 函式
from osuT5.osuT5.event import ContextType

class MapperEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 參數：7個 continuous + 1個 binary（super_timing）
        self.action_space = gym.spaces.Box(
            low=np.array([2.0, 3.0, 5.0, 0.8, 0.6, 1.0, 0.0]),  # 最後是 super_timing
            high=np.array([6.0, 10.0, 10.0, 2.0, 1.4, 10.0, 1.0]),
            dtype=np.float32
        )

        # 觀察空間這裡先簡化為空向量（黑盒）
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # 固定參數（可改成 init 傳入）
        self.input_audio = "audio.mp3"
        self.input_beatmap = "template.osu"
        self.output_dir = "output_maps"

        os.makedirs(self.output_dir, exist_ok=True)

    def reset(self):
        # 可以加一些隨機初始化邏輯
        return np.zeros((1,), dtype=np.float32)

    def step(self, action):
        # 解碼參數
        params = self._decode_action(action)

        # 跑 Mapperatorinator 並產生 .osu
        osu_path = self._generate_map(params)

        # 計算 reward
        reward = evaluate_osu_file(osu_path)

        # 單步問題，done 一律為 True
        return np.zeros((1,), dtype=np.float32), reward, True, {}

    def _decode_action(self, action):
        # 將連續空間值轉成實際 config 值
        return {
            "circle_size": float(action[0]),
            "overall_difficulty": float(action[1]),
            "approach_rate": float(action[2]),
            "slider_multiplier": float(action[3]),
            "temperature": float(action[4]),
            "cfg_scale": float(action[5]),
            "super_timing": bool(round(action[6])),
        }

    def _generate_map(self, params):
        # 建立臨時輸出檔案路徑
        tmp_dir = tempfile.mkdtemp(dir=self.output_dir)
        output_path = os.path.join(tmp_dir, "output.osu")

        # 建立 config.py（或直接呼叫 python script）
        config = {
            "audio_path": self.input_audio,
            "output_path": output_path,
            "beatmap_path": self.input_beatmap,
            "gamemode": "standard",
            "difficulty": 5,
            "mapper_id": None,
            "year": 2023,
            "hitsounded": True,
            "hp_drain_rate": 5,
            "keycount": 4,
            "hold_note_ratio": None,
            "scroll_speed_ratio": None,
            "descriptors": [],
            "negative_descriptors": [],
            "export_osz": False,
            "add_to_beatmap": False,
            "start_time": None,
            "end_time": None,
            "in_context": [ContextType(c.lower()) for c in "[NONE]".split(',')],
            "output_type": [ContextType(c.lower()) for c in "[MAP]".split(',')],
            "output_type": "osu",
            "seed": None,
            **params
        }

        # 實際呼叫 Mapperatorinator（依照你的介面）
        # 假設你有個 generate_from_config(config) 函式
        from Mapperatorinator.main import generate_from_config
        generate_from_config(config)

        return output_path