import torch
import torch.nn as nn


class TrajEncoder(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, output_dim=512):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "TrajEncoder"

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )

    def forward(self, traj_tensor: torch.Tensor) -> torch.Tensor:
        traj_tensor_emb = self.encoder(traj_tensor)
        traj_tensor_emb = torch.mean(traj_tensor_emb, dim=1)  # Average over the sequence length
        return traj_tensor_emb
