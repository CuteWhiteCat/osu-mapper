import json, os, glob

def build_rating_dict(osu_folder: str) -> dict[str, float]:
    """
    讀取 <id>.json → {"rating": xxx}，回傳 {<id>.osu: rating}
    """
    rating_dict = {}
    for json_path in glob.glob(os.path.join(osu_folder, "*.json")):
        beatmap_id = os.path.splitext(os.path.basename(json_path))[0]
        try:
            rating = json.load(open(json_path, "r", encoding="utf-8"))["rating"]
            rating_dict[f"{beatmap_id}.osu"] = float(rating)
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return rating_dict