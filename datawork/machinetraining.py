import lmdb
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

class LMDBChessDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(path, readonly=True, lock=False, readahead=True, max_readers=32)
        with self.env.begin() as txn:
            self.length = int(txn.stat()["entries"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        key = idx.to_bytes(8, 'big')
        with self.env.begin() as txn:
            val = txn.get(key)
        eval_f = struct.unpack(">f", val[:4])[0]
        eval_f = np.clip(eval_f, -100, 100) / 100.0
        arr = np.frombuffer(val, dtype=np.uint8, offset=4).reshape((12,8,8)).astype(np.float32)
        x = torch.tensor(arr)
        y = torch.tensor(eval_f, dtype=torch.float32)
        return x, y



class ChessEvalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(lmdb_path, epochs=2, batch_size=128, lr=1e-3, max_batches=500, model=None):
    dataset = LMDBChessDataset(lmdb_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if model is None:
        model = ChessEvalCNN()

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for i, (x, y) in enumerate(loader):
            # shape adjustment: CNN expects [batch, channels, H, W]
            x = x.float().to("cpu")
            y = y.float().to("cpu")

            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss={loss.item():.4f}")

            if max_batches and i >= max_batches:
                break

        print(f"Epoch {epoch+1} finished, Avg Loss={total_loss/(i+1):.4f}")

    torch.save(model.state_dict(), "chess_eval_cnn.pt")
    print("Training complete. Model saved to chess_eval_cnn.pt")

    return model


if __name__ == "__main__":
    model = train_model("chess_dataset.lmdb", epochs=1, batch_size=64, max_batches=5000)