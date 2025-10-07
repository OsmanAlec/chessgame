import numpy as np
import torch.nn as nn
import torch
import board

class ChessEvalCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(12, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*8*8, 256)
        self.fc2 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.flatten(x)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.tanh(x)
        return x.squeeze(-1)

class ChessAI:
    def __init__(self, model_path):
        self.model = ChessEvalCNN()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def evaluate(self, board):
        tensor = self.board_to_tensor(board)
        with torch.no_grad():
            score = self.model(tensor.unsqueeze(0)).item()
        return score  

    def evaluateMove(model, board_after_move: board):
        pieces = ['P', 'N', 'B', 'R', 'Q', 'K',
            'p', 'n', 'b', 'r', 'q', 'k']
        
        x = []
        for piece in pieces:
            current_layer = []
            for i in range(8):
                current_row = []
                for j in range(8):
                    current_row.append(1 if str(board_after_move.getPieceAt(i,j)) == piece else 0)
                current_layer.append(current_row)
            x.append(current_layer)
        
        # Convert to NumPy array
        x = np.array(x, dtype=np.float32)  # shape (12, 8, 8)
        
        # Convert to torch tensor and add batch dimension
        x_tensor = torch.tensor(x).unsqueeze(0)  # (1, 12, 8, 8)
        
        # Forward through model
        with torch.no_grad():
            score = model(x_tensor).item()

        return score
