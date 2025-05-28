import gym
import numpy as np
import tempfile
import os
from reward_function import evaluate_osu_file  # 你自訂的 reward 函式
from Mapperatorinator.osuT5.osuT5.event import ContextType
from dataclasses import dataclass, asdict
from Mapperatorinator.inference import main

@dataclass
class BaseConfig:
    audio_path: str
    output_path: str
    beatmap_path: str
    gamemode: str   = "standard"
    difficulty: int = 5
    mapper_id: int | None = None
    year: int  = 2023
    hitsounded: bool = True
    hp_drain_rate: int = 5
    keycount: int = 4
    hold_note_ratio: float | None = None
    scroll_speed_ratio: float | None = None
    descriptors: list[str] = ()
    negative_descriptors: list[str] = ()
    export_osz: bool = False
    add_to_beatmap: bool = False
    start_time: int | None = None
    end_time: int | None = None
    in_context: list[ContextType] = (ContextType.none,)
    output_type: str = "osu"
    seed: int | None = None

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

    def _generate_map(self, params: dict) -> str:
        """
        給 PPO 用：把動作轉好的 `params` 打進 Mapperatorinator，
        生成 .osu 檔並回傳路徑。
        """
        # ---------- 1. 準備輸出位置 ----------
        tmp_dir = tempfile.mkdtemp(dir=self.output_dir)
        output_osu = os.path.join(tmp_dir, "output.osu")

        # ---------- 2. 組裝 config ----------
        #   先用 dataclass 給一份「乾淨的基底」，之後再用 action 參數覆寫
        cfg = asdict(
            BaseConfig(
                audio_path   = self.input_audio,
                output_path  = output_osu,
                beatmap_path = self.input_beatmap,
            )
        )
        # 把 agent 產生的高階參數（CS/AR/Temperature …）塞進去
        cfg.update(params)

        # ---------- 3. 呼叫 Mapperatorinator ----------
        #   inference.main 會回傳 (cfg, result_path, osz_path)
        #   實際生成的 .osu 位置就會是 cfg["output_path"]
        try:
            _, _, _ = main(cfg)          # 只需 Side-effect：寫檔
        except Exception as e:
            # 若產生失敗就直接丟回 template，reward 會很低
            print(f"[MapperEnv] Mapperatorinator 失敗：{e}")
            return self.input_beatmap

        return output_osu