import torch
import torch.nn as nn

class CorrectionNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.Tanh()  # output small residuals [-1,1]
        )
        self.device = torch.device('cpu')
        self.to(self.device)
        # Todo if time: load pretrained weights or initialize optimizer

    def predict(self, state):
        #Inference only
        
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).to(self.device)
            out = self.net(x)
        return out.cpu().numpy()

