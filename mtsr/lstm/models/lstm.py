import torch


class BiLSTM_max(torch.nn.Module):
    def __init__(self, args):
        super(BiLSTM_max, self).__init__()

        self.num_layers = args.layers
        self.nSeries = args.nSeries
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.lstm_hidden = args.hidden
        self.dropout = args.dropout if self.num_layers > 1 else 0

        self.seq_len_x = args.seq_len_x
        self.out_seq_len = args.out_seq_len

        self.bilstm = torch.nn.LSTM(input_size=self.in_dim, hidden_size=self.lstm_hidden, bias=True,
                                    bidirectional=True, num_layers=self.num_layers, dropout=self.dropout)

        self.linear_1 = torch.nn.Linear(in_features=self.lstm_hidden * 2, out_features=self.lstm_hidden, bias=True)
        self.linear_2 = torch.nn.Linear(in_features=self.lstm_hidden, out_features=self.out_dim, bias=True)
        self.verbose = args.verbose

    def forward(self, input_tensor):
        # input x (b, seq_x, n, features)
        if self.verbose:
            print('input_tensor: ', input_tensor.size())
        b, s, n, f = input_tensor.size()
        # x = input_tensor.transpose(1, 2)
        # x = x.reshape(b * n, s, f)  # (bn, seq_x, f)
        x_out = []
        for i in range(n):
            x_in = input_tensor[:, :, i, :]
            out, _ = self.bilstm(x_in)  # (b, seq_x, hidden*2)
            out = self.linear_1(out[:, -1, :])  # (b, 1, n)
            out = torch.nn.functional.dropout(out, 0.2, training=self.training)
            out = self.linear_2(out)  # (bn, 1, 1)
            out = out.reshape(b, self.out_dim, 1)  # (b, 1, n)
            x_out.append(out)
        x_out = torch.stack(x_out, dim=2)
        x_out = x_out.squeeze(dim=-1)
        if self.verbose:
            print('x_out: ', x_out.size())

        return x_out
