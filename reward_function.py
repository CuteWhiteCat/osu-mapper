import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils import build_rating_dict
import matplotlib.pyplot as plt

OSU_FOLDER = "./osu_files"

# ==== 1. 解析 .osu 檔案 ====
def parse_osu_file(filepath):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    hit_objects = []
    in_hit_objects = False
    last_time = 0

    for line in lines:
        line = line.strip()
        if line.startswith("[HitObjects]"):
            in_hit_objects = True
            continue
        if in_hit_objects:
            if not line or line.startswith("["):
                break
            try:
                parts = line.split(",")
                if len(parts) < 5:
                    continue
                x = int(parts[0])
                y = int(parts[1])
                time = int(parts[2])
                object_type = int(parts[3]) & 0b1111

                time_delta = time - last_time if last_time != 0 else 0
                last_time = time

                hit_objects.append([
                    x / 512.0, y / 384.0, time_delta / 1000.0, object_type / 8.0
                ])
            except Exception as e:
                break
    return hit_objects

# ==== 2. LSTM 模型 ====
class BeatmapLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, dropout=0.3, bidirectional=True):
        super(BeatmapLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * self.num_directions, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output)
        return out.squeeze(-1)  # shape: (B, T)

# ==== 3. Dataset 與 DataLoader ====
class BeatmapDataset(torch.utils.data.Dataset):
    def __init__(self, osu_folder, rating_dict):
        self.filepaths = [
            os.path.join(osu_folder, f)
            for f in os.listdir(osu_folder)
            if f.endswith(".osu")
        ]
        self.rating_dict = rating_dict

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
    max_len = padded_seqs.size(1)
    padded_targets = torch.zeros((len(sequences), max_len))
    for i, (r, l) in enumerate(zip(ratings, lengths)):
        padded_targets[i, :l] = r  # dummy target
    return padded_seqs, torch.tensor(lengths), padded_targets

# ==== 4. 訓練 ====
def create_mask(lengths, max_len=None):
    if isinstance(lengths, list):
        lengths = torch.tensor(lengths)
    lengths = lengths.to(torch.long)
    if max_len is None:
        max_len = lengths.max()
    range_tensor = torch.arange(max_len).unsqueeze(0).to(lengths.device)
    lengths = lengths.unsqueeze(1)
    mask = (range_tensor < lengths).float()
    return mask

def train_model(dataset, num_epochs=200, lr=0.001):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
    model = BeatmapLSTM()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss(reduction='none')
    losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            inputs, lengths, targets = batch
            optimizer.zero_grad()
            outputs = model(inputs, lengths)  # shape: (B, T)
            loss = criterion(outputs, targets)  # shape: (B, T)
            mask = create_mask(lengths, max_len=outputs.size(1))  # shape: (B, T)
            masked_loss = loss * mask
            masked_loss = masked_loss.sum() / mask.sum()
            masked_loss.backward()
            optimizer.step()
            total_loss += masked_loss.item()

        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")

    plt.plot(losses, label='loss')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss Curve")
    plt.savefig("loss_curve.png")
    plt.close()

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
    seq = parse_osu_file(osu_path)
    if not seq:
        return 0.0
    lengths = torch.tensor([len(seq)], device=_device)
    tensor = torch.tensor([seq], dtype=torch.float32, device=_device)
    model = _load_model()
    scores = model(tensor, lengths)  # shape: (1, T)
    mask = create_mask(lengths, max_len=scores.size(1)).to(_device)
    avg_score = (scores * mask).sum() / mask.sum()
    return float(avg_score)

if __name__ == "__main__":
    rating_dict = build_rating_dict(OSU_FOLDER)
    dataset = BeatmapDataset(OSU_FOLDER, rating_dict)
    model = train_model(dataset)
    print("beatmap_model.pt 已生成，大小：", os.path.getsize("beatmap_model.pt"), "bytes")
