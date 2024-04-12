import torch.nn as nn

def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)

class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1, self.dim2 = dim1, dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(*self.shape)

class Norm(nn.Module):
    def __init__(self, norm_mode, num_channels, seq_len, d_model):
        super().__init__()
        self.norm_mode = norm_mode
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.d_model = d_model

        if norm_mode=="batch1d":
            self.norm = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        elif norm_mode=="batch2d":
            self.norm = nn.BatchNorm2d(num_channels)
        elif norm_mode=="layer":
            self.norm = nn.LayerNorm(d_model)
        else:
            raise ValueError("Please select a valid normalization mode: 'batch1d', 'batch2d', or 'layer'.")

    def forward(self, x):
        if self.norm_mode == "batch2d":
            batch_size = x.shape[0]
            x = x.view(batch_size, self.num_channels, self.seq_len, self.d_model)
            x = self.norm(x)
            x = x.view(batch_size*self.num_channels, self.seq_len, self.d_model)
            return x
        else:
            return self.norm(x)