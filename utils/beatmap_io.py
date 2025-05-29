from pathlib import Path
import json

def load_rating_dict(osu_root: Path) -> dict[int, float]:
    """讀 rating_table.json → {beatmapset_id: rating}"""
    table = osu_root / "rating_table.json"
    with table.open(encoding="utf-8") as fp:
        data = json.load(fp)
    return {int(k): float(v) for k, v in data.items()}
