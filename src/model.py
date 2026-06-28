import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2,1), (2,1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2,1), (2,1)),

            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU()
        )

        self.rnn = nn.LSTM(
            512,
            256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        conv = self.cnn(x)      # (B, 512, 1, W)
        conv = conv.squeeze(2)  # (B, 512, W)
        conv = conv.permute(0, 2, 1)  # (B, W, 512)

        rnn_out, _ = self.rnn(conv)
        output = self.fc(rnn_out)

        return output
