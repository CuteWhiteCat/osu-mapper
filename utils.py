import os, glob, json


def build_rating_dict(osu_folder: str) -> dict[str, float]:
    """
    根據 osu_folder 下的 rating_table.json 生成 {<beatmap_filename>: rating}。
    """
    rating_dict: dict[str, float] = {}
    table_path = os.path.join(osu_folder, "rating_table.json")
    if os.path.isfile(table_path):
        with open(table_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for beatmap_id, rating in data.items():
                rating_dict[f"{beatmap_id}.osu"] = float(rating)
        return rating_dict

    # fallback: parse individual JSON files that contain a 'rating' field
    for json_path in glob.glob(os.path.join(osu_folder, "*.json")):
        filename = os.path.basename(json_path)
        if filename.lower() == "rating_table.json":
            continue
        try:
            obj = json.load(open(json_path, "r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError, TypeError):
            continue
        if isinstance(obj, dict) and "rating" in obj:
            beatmap_id = os.path.splitext(filename)[0]
            try:
                rating_dict[f"{beatmap_id}.osu"] = float(obj["rating"])
            except (ValueError, TypeError):
                continue
    return rating_dict
