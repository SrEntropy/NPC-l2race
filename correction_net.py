import torch
import torch.nn as nn
import torch.optim as optim

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
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)
        self.loss_fn = nn.MSELoss()
    

    def predict(self, state):
        #Inference only
        with torch.no_grad():
            x = torch.tensor(state, dtype=torch.float32).to(self.device)
            out = self.net(x)
        return out.cpu().numpy()


    def train_step(self, batch):
        """
        batch: list of (state, target_residual) tuples
        """
        states, targets = zip(*batch)
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        targets = torch.tensor(targets, dtype=torch.float32).to(self.device)

        self.optimizer.zero_grad()
        preds = self.net(states)
        loss = self.loss_fn(preds, targets)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    

