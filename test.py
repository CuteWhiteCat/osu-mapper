from hydra import initialize, compose
import os, sys, types, pathlib, subprocess

# from google.colab import files

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

from Mapperatorinator.inference import main

with initialize(config_path="Mapperatorinator/configs", version_base="1.1"):
    cfg = compose(
        config_name="inference_v30",
        overrides=[
            "audio_path=audios/audio.mp3",
            "output_path=output_maps",
            "export_osz=True"
        ],
    )


_, result_path, osz_path = main(cfg)

if osz_path is not None:
    result_path = osz_path

download_path = (
    osz_path if (osz_path is not None and cfg.add_to_beatmap) else result_path
)

if download_path is None:
    raise RuntimeError("Beatmap 未生成，請檢查 cfg 參數是否正確")
print(f"Osu File locate at：{download_path}")
