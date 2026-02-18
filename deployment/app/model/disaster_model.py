import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class DisasterTwittsClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, num_layers, dropout, 
                 bidirectional=True, embedding=None, freeze_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        if embedding is not None:
            self.embedding.weight.data.copy_(embedding)
            self.embedding.weight.requires_grad = not freeze_embedding

        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim*(2 if bidirectional else 1), output_dim)

    def forward(self, x, lengths=None):
        x = self.embedding(x)
        if lengths is not None:
            packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed_x)
            lstm_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)

        if self.lstm.bidirectional:
            x = torch.cat([h_n[-2,:,:], h_n[-1,:,:]], dim=1)
        else:
            x = h_n[-1,:,:]

        x = self.dropout(x)
        out = self.fc(x).squeeze(1)
        return out
    
    @classmethod
    def from_pretrained(cls, config, vocab_size):
        '''
        Professional factory method to instantiate and load the model.
        '''
        # 1. Instantiate the class (cls refers to DisasterTwittsClassifier)
        model = cls(
            vocab_size=vocab_size,
            emb_dim=config.EMB_DIM,
            hidden_dim=config.HIDDEN_DIM,
            output_dim=config.OUTPUT_DIM,
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
            bidirectional=config.BIDIRECTIONAL,
            embedding=None,
            freeze_embedding=config.FREEZE_EMBEDDING
        )
        
        # 2. Load weights safely
        checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        
        # 3. Prepare for inference
        model.to(config.DEVICE)
        model.eval()
        
        return model
    
