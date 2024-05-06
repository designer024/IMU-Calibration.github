import torch
import torch.nn as nn

class IMUModel(torch.nn.Module):
    def __init__(self, 
                 encoder_hidden_size=512,
                 decoder_hidden_size=512,
                 encoder_layers=4,
                 decoder_layers=4,
                 encoder_max_length=256,
                 decoder_max_length=256,
                 vocab_size=18,
                 norm_first=False,
                 dropout=0.1,
                 initialize_method="uniform",
                 initialize_std=0.1,
                 device='cuda'):
        super().__init__()
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        assert self.encoder_hidden_size == self.decoder_hidden_size
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
        self.vocab_size=vocab_size
        self.dropout = dropout

        self.encoder_embedding = nn.Linear(self.vocab_size, self.encoder_hidden_size)
        self.decoder_embedding = nn.Linear(self.vocab_size, self.decoder_hidden_size)

        self.encoder_positional_embedding = nn.Parameter(data=torch.ones([1, self.encoder_max_length, self.encoder_hidden_size], dtype=torch.float32))
        self.decoder_positional_embedding_generator = nn.Embedding(self.decoder_max_length, self.decoder_hidden_size) 
        self.decoder_positional_embedding_seeds = torch.arange(self.decoder_max_length, device=device)

        self.final = nn.Conv1d(self.decoder_hidden_size, self.vocab_size, 1)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.encoder_hidden_size, nhead=8, batch_first=True, norm_first=norm_first, dropout=dropout),
            num_layers=self.encoder_layers
        )
        self.decoder = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=self.decoder_hidden_size, nhead=8, batch_first=True,norm_first=norm_first, dropout=dropout)
            for _ in range(self.decoder_layers)
        ])

        self.initialize(initialize_method, initialize_std)

    def initialize(self, init_method, std):
        if init_method == "uniform":
            initrange = std
            self.decoder_positional_embedding_generator.weight.data.uniform_(-initrange,initrange)
            torch.nn.init.uniform_(self.encoder_positional_embedding, a=-initrange, b=initrange)
        elif init_method == "normal":
            self.decoder_positional_embedding_generator.weight.data.normal_(std=std)
            torch.nn.init.normal_(self.encoder_positional_embedding, mean=0.0,std=std)
        elif init_method == "trunc_normal":
            torch.nn.init.trunc_normal_(self.decoder_positional_embedding_generator.weight, std=std,a=-2*std,b=2*std)
            torch.nn.init.trunc_normal_(self.encoder_positional_embedding, mean=0.0,std=std,a=-2*std,b=2*std)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, tgt):
        """
        params
        src: (B, F, J, C)
        tgt: (B, F, J, C)
        """
        b, f, j, c = src.shape
        encoded = self.encoder_embedding(src.view(b, f, -1))
        encoded += self.encoder_positional_embedding

        tgt_in = self.decoder_embedding(tgt.view(b, f, -1))
        target_length = tgt.shape[1]
        tgt_in += self.decoder_positional_embedding_generator(self.decoder_positional_embedding_seeds[:target_length]).unsqueeze(0)
        contexts = self.encoder(encoded)
        decoder_output = tgt_in
        square_mask = self.generate_square_subsequent_mask(target_length).to(tgt.device)
        for n, decoder_layer in enumerate(self.decoder):
            decoder_output = decoder_layer(tgt=decoder_output, memory=contexts, tgt_mask=square_mask)
        
        pred = self.final(decoder_output.permute(0,2,1)).permute((0,2,1)).view(b, f, j, c)
        return pred

if __name__ == '__main__':
    model = IMUModel(device='cpu')
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    print('model size:', model_params)
    x = torch.zeros(32,256,2,9).float()
    tgt = torch.zeros(32,256,2,9).float()
    y = model(x, tgt)
    print(y.shape)