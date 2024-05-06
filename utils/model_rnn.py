import torch
import torch.nn as nn


class IMUModel(torch.nn.Module):
    def __init__(self, num_joints, in_features, embed_dim=512, hidden_size=512,
            num_layers=2, dropout=0.25, bidirectional=True, max_len=512):
        super().__init__()
        
        # arch
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # model
        self.expand_conv = nn.Linear(num_joints*in_features, self.embed_dim)
        
        self.gru = nn.GRU(self.embed_dim, self.hidden_size, self.num_layers, dropout=dropout,
                        bidirectional=self.bidirectional, batch_first=True)
        self.drop_layout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.encoder_output_size, num_joints*in_features, bias=False)
        
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.h0_num_layers, batch_size, self.hidden_size).zero_()
        return hidden

    @property
    def encoder_output_size(self):
        return 2 * self.hidden_size if self.bidirectional else self.hidden_size
    
    @property
    def h0_num_layers(self):
        return 2 * self.num_layers if self.bidirectional else self.num_layers

    def forward(self, x, h_0=None):
        b, f, j, c = x.shape
        x = x.contiguous().view(b, f, -1) # (b, f, j*c)
        
        x = self.drop(self.act(self.expand_conv(x)))
        self.gru.flatten_parameters()
        if h_0 != None:
            x, h_n = self.gru(x, h_0)
        else:
            x, h_n = self.gru(x)
        
        pred = self.fc(self.drop_layout(self.act(x)))
        return pred.view(b, f, j, c)

if __name__ == '__main__':
    model = BoneLengthModel(17, 2)
    
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('Trainable parameter count:', model_params)
    
    A = torch.zeros(64,81,17,2)
    B = model(A)
    print(B.shape)