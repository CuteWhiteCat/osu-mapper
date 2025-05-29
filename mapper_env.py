import sys
import types
import pathlib
import os
import zipfile
import tempfile

import gym
import numpy as np
from dataclasses import dataclass, asdict, field
from omegaconf import OmegaConf

# ---- mapper_env.py：最前面就寫 ----
import sys, types, pathlib

_repo = pathlib.Path(__file__).resolve().parent / "Mapperatorinator"
if _repo.exists():
    # 1) 讓  import Mapperatorinator.*  能找到
    sys.path.insert(0, str(_repo))

    # 2) 為  osuT5.*  建立「虛擬」頂層 package
    osuT5_root = _repo / "osuT5"
    if osuT5_root.exists():
        sys.path.insert(0, str(osuT5_root))  # 讓次層模組可搜尋
        if "osuT5" not in sys.modules:  # 動態註冊頂層 package
            pkg = types.ModuleType("osuT5")
            pkg.__path__ = [str(osuT5_root)]
            sys.modules["osuT5"] = pkg
# ------------------------------------

from Mapperatorinator.inference import main
from Mapperatorinator.osuT5.osuT5.event import ContextType
from reward_function import evaluate_osu_file

osu_folder = "osu_files"


@dataclass
class BaseConfig:
    audio_path: str
    output_path: str
    beatmap_path: str
    model_path: str
    device: str = "cpu"
    osut5: dict = field(default_factory=dict)
    gamemode: str = "standard"
    difficulty: int = 5
    mapper_id: int | None = None
    year: int = 2023
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
    in_context: list[ContextType] = (ContextType.NONE,)
    output_type: str = "osu"
    seed: int | None = None


osu_folder = "osu_files"


class MapperEnv(gym.Env):
    def __init__(self):
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

        # 固定參數（可改成 init 傳入）
        self.input_audio = "audio.mp3"
        # self.input_beatmap = "template.osu"
        self.input_beatmap = os.path.join(os.getcwd(), "osu_files", "template.osu")
        template = None

        # 1) 先找 .osu
        for fn in os.listdir(osu_folder):
            if fn.lower().endswith(".osu"):
                template = os.path.join(osu_folder, fn)
                break

        # 2) 若沒找到，再 unzip 第一個 .osz
        if template is None:
            osz = next(
                (f for f in os.listdir(osu_folder) if f.lower().endswith(".osz")), None
            )
            if osz:
                with zipfile.ZipFile(os.path.join(osu_folder, osz), "r") as z:
                    osu_members = [
                        m for m in z.namelist() if m.lower().endswith(".osu")
                    ]
                    if osu_members:
                        z.extract(osu_members[0], osu_folder)
                        template = os.path.join(osu_folder, osu_members[0])

        # 3) 還是沒找到就報錯
        if template is None or not os.path.isfile(template):
            raise FileNotFoundError(f"No .osu template found in {osu_folder}")

        # 最後把找到的路徑指回 self.input_beatmap
        self.input_beatmap = template

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
        # 1) 準備輸出目錄
        tmp_dir = tempfile.mkdtemp(dir=self.output_dir)
        output_osu = os.path.join(tmp_dir, "output.osu")
        # 2) 組 config
        cfg = asdict(BaseConfig(
            audio_path=self.input_audio,
            output_path=output_osu,
            beatmap_path=self.input_beatmap,
            # 目前預設是空或 HF repo，改成本地路徑：
        ))
        cfg.update(params)
        # 3) 呼叫 Mapperatorinator
        try:
            # 僅呼叫一次、傳入 OmegaConf 物件
            _ = main(OmegaConf.create(cfg))
        except Exception as e:
            print(f"[MapperEnv] Mapperatorinator 失敗：{e}")
            return self.input_beatmap
        return output_osu
