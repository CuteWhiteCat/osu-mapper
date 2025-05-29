import sys
import types
import pathlib
import os
import zipfile
import tempfile

from typing import Any, Optional

import gym
import numpy as np
from dataclasses import dataclass, asdict, field
from omegaconf import OmegaConf, MISSING
from hydra.core.config_store import ConfigStore
from Mapperatorinator.osuT5.osuT5.event import ContextType


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
class SpectrogramConfig:
    implementation: str = "nnAudio"  # Spectrogram implementation (nnAudio/torchaudio)
    log_scale: bool = False
    sample_rate: int = 16000
    hop_length: int = 128
    n_fft: int = 1024
    n_mels: int = 388
    f_min: int = 0
    f_max: int = 8000
    pad_mode: str = "constant"


@dataclass
class ModelConfig:
    name: str = "openai/whisper-base"  # Model name
    config_base: str = ""  # Model base for config lookup
    input_features: bool = True
    project_encoder_input: bool = True
    embed_decoder_input: bool = True
    manual_norm_weights: bool = False
    do_style_embed: bool = False
    do_difficulty_embed: bool = False
    do_mapper_embed: bool = False
    do_song_position_embed: bool = False
    cond_dim: int = 128
    cond_size: int = 0
    rope_type: str = "dynamic"  # RoPE type (dynamic/static)
    rope_encoder_scaling_factor: float = 1.0
    rope_decoder_scaling_factor: float = 1.0
    spectrogram: SpectrogramConfig = field(default_factory=SpectrogramConfig)
    overwrite: dict = field(default_factory=lambda: {})  # Overwrite model config
    add_config: dict = field(default_factory=lambda: {})  # Add to model config


@dataclass
class DataConfig:
    dataset_type: str = "mmrs"   # Dataset type (ors/mmrs)
    train_dataset_path: str = "/workspace/datasets/MMRS39389"  # Training dataset directory
    train_dataset_start: int = 0  # Training dataset start index
    train_dataset_end: int = 38689  # Training dataset end index
    test_dataset_path: str = "/workspace/datasets/MMRS39389"  # Testing/validation dataset directory
    test_dataset_start: int = 38689  # Testing/validation dataset start index
    test_dataset_end: int = 39389  # Testing/validation dataset end index
    src_seq_len: int = 1024
    tgt_seq_len: int = 2048
    sample_rate: int = 16000
    hop_length: int = 128
    cycle_length: int = 16
    per_track: bool = True  # Loads all beatmaps in a track sequentially which optimizes audio data loading
    only_last_beatmap: bool = False  # Only use the last beatmap in the mapset
    center_pad_decoder: bool = False  # Center pad decoder input
    num_classes: int = 152680
    num_diff_classes: int = 24  # Number of difficulty classes
    max_diff: int = 12  # Maximum difficulty of difficulty classes
    num_cs_classes: int = 21  # Number of circle size classes
    class_dropout_prob: float = 0.2
    diff_dropout_prob: float = 0.2
    mapper_dropout_prob: float = 0.2
    cs_dropout_prob: float = 0.2
    year_dropout_prob: float = 0.2
    hold_note_ratio_dropout_prob: float = 0.2
    scroll_speed_ratio_dropout_prob: float = 0.2
    descriptor_dropout_prob: float = 0.2
    add_gamemode_token: bool = True
    add_diff_token: bool = True
    add_style_token: bool = False
    add_mapper_token: bool = True
    add_cs_token: bool = True
    add_year_token: bool = True
    add_hitsounded_token: bool = True  # Add token for whether the map has hitsounds
    add_song_length_token: bool = True  # Add token for the length of the song
    add_song_position_token: bool = True  # Add token for the position of the song in the mapset
    add_descriptors: bool = True
    add_empty_sequences: bool = True
    add_empty_sequences_at_step: int = -1
    add_pre_tokens: bool = False
    add_pre_tokens_at_step: int = -1
    max_pre_token_len: int = -1
    timing_random_offset: int = 2
    add_gd_context: bool = False  # Prefix the decoder with tokens of another beatmap in the mapset
    min_difficulty: float = 0  # Minimum difficulty to consider including in the dataset
    sample_weights_path: str = ''  # Path to sample weights
    rhythm_weight: float = 3.0  # Weight of rhythm tokens in the loss calculation
    lookback: float = 0  # Fraction of audio sequence to fill with tokens from previous inference window
    lookahead: float = 0  # Fraction of audio sequence to skip at the end of the audio window
    context_types: list[dict[str, list[ContextType]]] = field(default_factory=lambda: [
        {"in": [ContextType.NONE], "out": [ContextType.TIMING, ContextType.KIAI, ContextType.MAP, ContextType.SV]},
        {"in": [ContextType.NO_HS], "out": [ContextType.TIMING, ContextType.KIAI, ContextType.MAP, ContextType.SV]},
        {"in": [ContextType.GD], "out": [ContextType.TIMING, ContextType.KIAI, ContextType.MAP, ContextType.SV]}
    ])  # List of context types to include in the dataset
    context_weights: list[float] = field(default_factory=lambda: [4, 1, 1])  # List of weights for each context type. Determines how often each context type is sampled
    descriptors_path: str = ''  # Path to file with all beatmap descriptors
    mappers_path: str = ''  # Path to file with all beatmap mappers
    add_timing: bool = False  # Add beatmap timing to map context
    add_out_context_types: bool = True  # Add tokens indicating types of the out context
    add_snapping: bool = True  # Model hit object snapping
    add_timing_points: bool = True  # Model beatmap timing with timing points
    add_hitsounds: bool = True  # Model beatmap hitsounds
    add_distances: bool = True  # Model hit object distances
    add_positions: bool = True  # Model hit object coordinates
    position_precision: int = 32  # Precision of hit object coordinates
    position_split_axes: bool = False  # Split hit object X and Y coordinates into separate tokens
    position_range: list[int] = field(default_factory=lambda: [-256, 768, -256, 640])  # Range of hit object coordinates
    dt_augment_prob: float = 0.5  # Probability of augmenting the dataset with DT
    dt_augment_range: list[float] = field(default_factory=lambda: [1.25, 1.5])  # Range of DT augmentation
    types_first: bool = True  # Put the type token at the start of the group before the timeshift token
    add_kiai: bool = True  # Add kiai times to map context
    gamemodes: list[int] = field(default_factory=lambda: [0, 1, 2, 3])  # List of gamemodes to include in the dataset
    mania_bpm_normalized_scroll_speed: bool = True  # Normalize mania scroll speed by BPM
    add_sv_special_token: bool = True  # Add extra special token for current SV
    add_sv: bool = True  # Model slider velocity in std and ctb
    add_mania_sv: bool = False  # Add mania scroll velocity in map context


@dataclass
class DataloaderConfig:
    num_workers: int = 8


@dataclass
class OptimizerConfig:  # Optimizer settings
    name: str = "adamwscale"  # Optimizer
    base_lr: float = 1e-2
    batch_size: int = 128  # Batch size per GPU
    total_steps: int = 65536
    warmup_steps: int = 10000
    lr_scheduler: str = "cosine"
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    grad_acc: int = 8
    final_cosine: float = 1e-5


@dataclass
class EvalConfig:
    every_steps: int = 1000
    steps: int = 500


@dataclass
class CheckpointConfig:
    every_steps: int = 5000


@dataclass
class LoggingConfig:
    log_with: str = 'wandb'     # Logging service (wandb/tensorboard)
    every_steps: int = 10
    grad_l2: bool = True
    weights_l2: bool = True
    mode: str = 'online'


@dataclass
class ProfileConfig:
    do_profile: bool = False
    early_stop: bool = False
    wait: int = 8
    warmup: int = 8
    active: int = 8
    repeat: int = 1


@dataclass
class TrainConfig:
    compile: bool = True
    device: str = "gpu"
    precision: str = "bf16"
    seed: int = 42
    flash_attention: bool = False
    checkpoint_path: str = ""
    pretrained_path: str = ""
    pretrained_t5_compat: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    hydra: Any = MISSING
    mode: str = "train"

try:
    OmegaConf.register_new_resolver("context_type", lambda x: ContextType(x.lower()))
except ValueError:
    # already registered
    pass

cs = ConfigStore.instance()
cs.store(group="osut5", name="base_train", node=TrainConfig)


@dataclass
class BaseConfig:
    audio_path: str
    output_path: str
    beatmap_path: str
    model_path: str
    diff_ckpt: str = "OliBomby/osu-diffusion-v2"
    version: str = "Mapperatorinator V30"
    device: str = "cpu"
    osut5: dict = field(default_factory=lambda: asdict(TrainConfig()))    
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
    in_context: list[str] = field(default_factory=list)
    output_type: list[str] = field(default_factory=lambda: ["MAP"])
    # output_type: str = "osu"
    seed: int | None = None
    temperature: float = 0.9            
    top_p: float = 0.9
    generate_positions: bool = False
    timesteps: list[int] = field(default_factory=lambda: [10] + [0] * 99)
    beatmap_id: int | None = None
    slider_tick_rate: float | None = None
    lookback: float = 0.5
    lookahead: float = 0.4 
    timing_leniency: int = 20  
    cfg_scale: float = 1.0  
    temperature: float = 1.0 
    timing_temperature: float = 0.1 
    mania_column_temperature: float = 0.5  
    taiko_hit_temperature: float = 0.5 
    timeshift_bias: float = 0.0 
    top_p: float = 0.95 
    top_k: int = 0 
    parallel: bool = False 
    do_sample: bool = True 
    num_beams: int = 1
    super_timing: bool = False 
    timer_num_beams: int = 2 
    timer_bpm_threshold: float = 0.7  
    timer_cfg_scale: float = 1.0 
    timer_iterations: int = 20 
    max_batch_size: int = 16 
    ###
    diff_cfg_scale: float = 1.0  # Scale of classifier-free guidance
    compile: bool = False  # PyTorch 2.0 optimization
    pad_sequence: bool = False  # Pad sequence to max_seq_len
    diff_ckpt: str = ''  # Path to checkpoint for diffusion model
    diff_refine_ckpt: str = ''  # Path to checkpoint for refining diffusion model
    beatmap_idx: str = 'osu_diffusion/beatmap_idx.pickle'  # Path to beatmap index
    refine_iters: int = 10  # Number of refinement iterations
    random_init: bool = False  # Whether to initialize with random noise instead of positions generated by the previous model
    timesteps: list[int] = field(default_factory=lambda: [100, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # The number of timesteps we want to take from equally-sized portions of the original process
    max_seq_len: int = 1024  # Maximum sequence length for diffusion
    overlap_buffer: int = 128  # Buffer zone at start and end of sequence to avoid edge effects (should be less than half of max_seq_len)
    # Metadata settings
    bpm: int = 120  # Beats per minute of input audio
    offset: int = 0  # Start of beat, in miliseconds, from the beginning of input audio
    title: str = ''  # Song title
    artist: str = ''  # Song artist
    creator: str = ''  # Beatmap creator
    version: str = ''  # Beatmap version
    background: Optional[str] = None  # File name of background image
    preview_time: int = -1  # Time in milliseconds to start previewing the song

    # 我不知道在哪(TrainConfig)
    flash_attention: bool = False
    checkpoint_path: str = ""
    pretrained_path: str = ""
    pretrained_t5_compat: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    hydra: Any = MISSING
    mode: str = "train"

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
        self.model_path = str(pathlib.Path(__file__).resolve().parent / "beatmap_model.pt")
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
            model_path="OliBomby/Mapperatorinator-v30",
        ))
        cfg.update(params)
        config_dir = pathlib.Path(__file__).resolve().parent / "configs"
        config_dir.mkdir(exist_ok=True)
        cfg_file = config_dir / "inference.yaml"
        OmegaConf.save(config=OmegaConf.create(cfg), f=str(cfg_file))
        # 3) 呼叫 Mapperatorinator
        try:
            # 僅呼叫一次、傳入 OmegaConf 物件
            _ = main(OmegaConf.create(cfg))
        except Exception as e:
            print(f"[MapperEnv] Mapperatorinator 失敗：{e}")
            return self.input_beatmap
        return output_osu
