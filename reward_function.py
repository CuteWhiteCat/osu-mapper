import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import build_rating_dict

OSU_FOLDER = "./osu_files"

# ==== 1. 解析 .osu 檔案 ====

def parse_osu_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    hit_objects = []
    in_hit_objects = False
    last_time = 0

    for line in lines:
        line = line.strip()
        if line.startswith('[HitObjects]'):
            in_hit_objects = True
            continue
        if in_hit_objects:
            if not line or line.startswith('['):
                break
            parts = line.split(',')
            if len(parts) < 5:
                continue
            x = int(parts[0])
            y = int(parts[1])
            time = int(parts[2])
            object_type = int(parts[3]) & 0b1111  # Only keep object type flags

            time_delta = time - last_time if last_time != 0 else 0
            last_time = time

            hit_objects.append([x / 512.0, y / 384.0, time_delta / 1000.0, object_type / 8.0])  # normalize

    return hit_objects

# ==== 2. LSTM 模型 ====

class BeatmapLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2):
        super(BeatmapLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed)
        out = self.fc(hn[-1])
        return out.squeeze(1)

# ==== 3. Dataset 與 DataLoader ====

class BeatmapDataset(torch.utils.data.Dataset):
    def __init__(self, osu_folder, rating_dict):
        self.filepaths = [os.path.join(osu_folder, f) for f in os.listdir(osu_folder) if f.endswith('.osu')]
        self.rating_dict = rating_dict  # filename -> rating

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        path = self.filepaths[idx]
        filename = os.path.basename(path)
        sequence = parse_osu_file(path)
        rating = self.rating_dict.get(filename, 0.0)
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(rating, dtype=torch.float32)

def collate_fn(batch):
    sequences, ratings = zip(*batch)
    lengths = [len(seq) for seq in sequences]
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    return padded_seqs, torch.tensor(lengths), torch.tensor(ratings)

# ==== 4. 訓練 ====

def train_model(dataset, num_epochs=20, lr=0.001):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    model = BeatmapLSTM()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs, lengths, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs, lengths)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "beatmap_model.pt")

    return model

# ==== 5. 推論 API ====
_loaded_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model():
    global _loaded_model
    if _loaded_model is None:
        m = BeatmapLSTM().to(_device)
        m.load_state_dict(torch.load("beatmap_model.pt", map_location=_device))
        m.eval()
        _loaded_model = m
    return _loaded_model

@torch.no_grad()
def evaluate_osu_file(osu_path: str) -> float:
    """
    給環境呼叫的 reward 函式：回傳預測 rating
    """
    seq = parse_osu_file(osu_path)
    if not seq:
        return 0.0     # 空圖給 0
    lengths = torch.tensor([len(seq)], device=_device)
    tensor = torch.tensor([seq], dtype=torch.float32, device=_device)
    model = _load_model()
    score = model(tensor, lengths).item()
    return float(score)

if __name__ == '__main__':
    rating_dict = build_rating_dict(OSU_FOLDER)
    dataset = BeatmapDataset('./osu_files', rating_dict)
    model = train_model(dataset)
    # train_model 内已经 save 了，至此模型文件就生成在当前目录
    print("beatmap_model.pt 已生成，大小：", os.path.getsize("beatmap_model.pt"), "bytes")