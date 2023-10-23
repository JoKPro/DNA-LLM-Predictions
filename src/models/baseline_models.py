import torch
from torch import nn
import pytorch_lightning as pl


class LogisticRegression(pl.LightningModule): 
    """Logistic Regression Model for Binary Classification.
    """
    def __init__(self, input_dim): 
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        self.criterion = nn.BCELoss()

    def forward(self, x): 
        outputs = torch.sigmoid(self.linear(x))
        return outputs

    def training_step(self, batch, batch_idx): 
        x, y = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss

    def configure_optimizers(self): 
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
