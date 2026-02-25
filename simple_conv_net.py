import torch
import torch.nn as nn


class SimpleConvNet(nn.Module):
    """Простая CNN для бинарной классификации (gamma/proton)."""

    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=20, out_channels=15, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(9 * 9 * 15, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 1),
        )

        self.to(self.device)

        # BCEWithLogitsLoss более численно стабильна, чем Sigmoid + BCELoss.
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Обратная совместимость с текущим ноутбуком.
        self.my_loss_fn = self.loss_fn
        self.my_learning_rate = self.learning_rate
        self.my_optimizer = self.optimizer

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x):
        """Вернуть вероятности класса gamma."""
        return torch.sigmoid(self.forward(x))
