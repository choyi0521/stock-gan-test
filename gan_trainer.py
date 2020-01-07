import torch
import torch.nn
from torch_trainer import TorchTrainer
from models.recurrent_models import LSTMGenerator, LSTMDiscriminator
from models.convolutional_models import TCNGenerator, TCNDiscriminator
from torch.utils.data import DataLoader


class LSTMGANTrainer(TorchTrainer):
    def __init__(self, n_epochs, batch_size, noise_dim, etf_dataset, num_workers=1, model='TCN'):
        super().__init__()

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.dataloader = DataLoader(dataset=etf_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.noise_dim = noise_dim

        # models
        assert model == 'LSTM' or model == 'TCN'
        if model == 'LSTM':
            lcond_dim = 6
            hidden_dim = 128
            n_layers = 3
            self.generator = LSTMGenerator(
                noise_dim=noise_dim,
                output_dim=lcond_dim,
                hidden_dim=hidden_dim,
                lcond_dim=lcond_dim,
                gcond_dim=1,
                n_layers=n_layers
            ).to(self.device)
            self.discriminator = LSTMDiscriminator(
                input_dim=lcond_dim,
                hidden_dim=hidden_dim,
                lcond_dim=lcond_dim,
                gcond_dim=1,
                n_layers=n_layers
            ).to(self.device)
        elif model == 'TCN':
            lcond_dim = 6
            hidden_dim = 32#128
            n_layers = 8
            self.generator = TCNGenerator(
                noise_dim=noise_dim,
                output_dim=lcond_dim,
                hidden_dim=hidden_dim,
                lcond_dim=lcond_dim,
                gcond_dim=1,
                n_layers=n_layers
            ).to(self.device)
            self.discriminator = TCNDiscriminator(
                input_dim=lcond_dim,
                hidden_dim=hidden_dim,
                lcond_dim=lcond_dim,
                gcond_dim=1,
                n_layers=n_layers
            ).to(self.device)

        # criterion
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

        # optimizers
        self.optimizer_g = torch.optim.Adam(self.generator.parameters())
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters())

    def train(self):
        self.generator.train()
        self.discriminator.train()
        for epoch in range(self.n_epochs):
            for i, data in enumerate(self.dataloader):
                lcond, gcond, target = data
                lcond = lcond.to(self.device)
                gcond = gcond.to(self.device)
                target = target.to(self.device)

                z = torch.randn((self.batch_size, lcond.shape[1], self.noise_dim), device=self.device)
                output = self.generator(z, lcond, gcond)
                fake_label = torch.zeros(lcond.shape[:2], device=self.device)
                real_label = torch.ones(lcond.shape[:2], device=self.device)

                # Update discriminator
                self.optimizer_d.zero_grad()
                real_loss = self.criterion(self.discriminator(target, lcond, gcond), real_label)
                fake_loss = self.criterion(self.discriminator(output.detach(), lcond, gcond), fake_label)
                d_loss = (real_loss+fake_loss) / 2
                d_loss.backward()
                self.optimizer_d.step()

                # Update generator
                self.optimizer_g.zero_grad()
                g_loss = self.criterion(self.discriminator(output, lcond, gcond), real_label)
                g_loss.backward()
                self.optimizer_g.step()

                if i % 10 == 0:
                    print(i)

    def validate(self):
        with torch.no_grad():
            self.generator.eval()
            self.discriminator.eval()

    def profile(self):
        import torchvision.models as models

        model = models.densenet121(pretrained=True)
        x = torch.randn((1, 3, 224, 224), requires_grad=True)

        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            model(x)

        print(prof)


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from pandas_datareader.data import DataReader
    from datetime import datetime

    etfs = ['VTI', 'EFA', 'EEM', 'TLT', 'TIP', 'VNQ']
    train_start = datetime(2005, 1, 1)
    train_end = datetime(2018, 12, 31)
    test_start = datetime(2019, 1, 1)
    test_end = datetime(2019, 12, 31)
    train = DataReader(etfs, 'yahoo', start=train_start, end=train_end)['Adj Close']
    test = DataReader(etfs, 'yahoo', start=test_start, end=test_end)['Adj Close']

    from preprocessor import ETFScaler
    from etf_dataset import ETFDataset
    train_data = train.values
    max_pred_steps = 200
    scaler = ETFScaler(train_data, max_pred_steps)
    etf_dataset = ETFDataset(etfs=train_data, seq_len=2000, max_pred_steps=max_pred_steps, scaler=scaler)
    print('length:', len(etf_dataset))

    n_epochs = 10
    batch_size = 16
    noise_dim = 4#16
    lgt = LSTMGANTrainer(n_epochs=n_epochs,
                         batch_size=batch_size,
                         noise_dim=noise_dim,
                         etf_dataset=etf_dataset,
                         num_workers=1,
                         model='TCN'
                         )
    #lgt.profile()
    lgt.train()

