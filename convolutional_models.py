import torch
import torch.nn as nn
from modules.tcn import TemporalConvNet


class TCNGenerator(nn.Module):
    def __init__(self,
                 noise_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 lcond_dim: int = 0,
                 gcond_dim: int = 0,
                 n_layers: int = 8,
                 kernel_size: int = 2,
                 dropout: float = 0.2
                 ):
        """
        Convolutional generator
        :param noise_dim: noise dimension
        :param output_dim: output dimension
        :param hidden_dim: hidden dimension
        :param lcond_dim: local condition dimension
        :param gcond_dim: global condition dimension
        :param n_layers: the number of layers
        :param kernel_size: the size of kernel
        :param dropout: dropout ratio
        """
        super().__init__()
        self.lcond_dim = lcond_dim
        self.gcond_dim = gcond_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.tcn = TemporalConvNet(noise_dim+lcond_dim+gcond_dim,
                                   [hidden_dim]*n_layers,
                                   kernel_size=kernel_size,
                                   dropout=dropout
                                   )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, noise, local_condition=None, global_condition=None):
        """
        :param noise: noise tensor of shape (batch_size, seq_len, noise_dim)
        :param local_condition: local condition tensor of shape (batch_size, seq_len, lcond_dim)
        :param global_condition: global condition tensor of shape (batch_size, gcond_dim)
        :return: Output tensor of shape (batch_size, seq_len, output_dim)
        """
        b, t, c = noise.size()
        if self.lcond_dim > 0:
            input = torch.cat((noise, local_condition), axis=2)
        if self.gcond_dim > 0:
            input = torch.cat((input, global_condition.unsqueeze(1).expand(b, t, self.gcond_dim)), axis=2)
        output = self.tcn(input.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output)
        return output


class TCNDiscriminator(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 lcond_dim: int = 0,
                 gcond_dim: int = 0,
                 n_layers: int = 8,
                 kernel_size: int = 2,
                 dropout: float = 0.2
                 ):
        """
        Convolutional discriminator
        :param input_dim: input dimension
        :param hidden_dim: hidden dimension
        :param lcond_dim: local condition dimension
        :param gcond_dim: global condition dimension
        :param n_layers: the number of layers
        :param kernel_size: the size of kernel
        :param dropout: dropout ratio
        """
        super().__init__()
        self.lcond_dim = lcond_dim
        self.gcond_dim = gcond_dim
        self.hidden_dim = hidden_dim

        self.tcn = TemporalConvNet(input_dim+lcond_dim+gcond_dim,
                                   [hidden_dim]*n_layers,
                                   kernel_size=kernel_size,
                                   dropout=dropout
                                   )
        self.linear = nn.Linear(hidden_dim, 1)

    def forward(self, input, local_condition=None, global_condition=None):
        """
        :param input: Input tensor of shape (batch_size, seq_len, input_dim)
        :param local_condition: local condition tensor of shape (batch_size, seq_len, lcond_dim)
        :param global_condition: global condition tensor of shape (batch_size, gcond_dim)
        :return: Output tensor of shape (batch_size, seq_len)
        """
        b, t, c = input.size()
        if self.lcond_dim > 0:
            input = torch.cat((input, local_condition), axis=2)
        if self.gcond_dim > 0:
            input = torch.cat((input, global_condition.unsqueeze(1).expand(b, t, self.gcond_dim)), axis=2)

        output = self.tcn(input.transpose(1, 2)).transpose(1, 2)
        output = self.linear(output).view(b, t)
        return output


if __name__ == '__main__':
    noise_dim = 128
    hidden_dim = 256
    input_dim=output_dim = 256
    lcond_dim = 64
    gcond_dim = 64
    batch_size = 32
    seq_len = 300
    n_layers = 3

    g = LSTMGenerator(noise_dim=noise_dim, output_dim=output_dim, hidden_dim=hidden_dim, lcond_dim=lcond_dim, gcond_dim=gcond_dim, n_layers=n_layers)
    d = LSTMDiscriminator(input_dim=input_dim, hidden_dim=hidden_dim, lcond_dim=lcond_dim, gcond_dim=gcond_dim, n_layers=n_layers)

    noise = torch.randn((batch_size, seq_len, noise_dim))
    lcond = torch.zeros((batch_size, seq_len, lcond_dim))
    gcond = torch.zeros((batch_size, gcond_dim))
    output = g(noise, lcond, gcond)
    print('generator output size', output.size())
    output = d(output, lcond, gcond)
    print('discriminator output size', output.size())