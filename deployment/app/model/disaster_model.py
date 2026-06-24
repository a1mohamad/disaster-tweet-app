import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class DisasterTwittsClassifier(nn.Module):
    """LSTM classifier architecture used by the production predictor."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool = True,
        embedding: torch.Tensor | None = None,
        freeze_embedding: bool = True,
    ) -> None:
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

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Run padded token ids through embedding, LSTM, dropout, and output layer."""
        x = self.embedding(x)
        if lengths is not None:
            packed_x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (h_n, _) = self.lstm(packed_x)
        else:
            _, (h_n, _) = self.lstm(x)

        if self.lstm.bidirectional:
            x = torch.cat([h_n[-2,:,:], h_n[-1,:,:]], dim=1)
        else:
            x = h_n[-1,:,:]

        x = self.dropout(x)
        out = self.fc(x).squeeze(1)
        return out
    
    @classmethod
    def from_pretrained(cls, config: object, vocab_size: int) -> "DisasterTwittsClassifier":
        """Instantiate the classifier and load trained weights from disk."""
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
        
        if config.DEVICE == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(config.DEVICE)

        # Support both raw state_dict files and trainer checkpoints.
        checkpoint = torch.load(config.MODEL_PATH, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        model.device = device
        
        return model
    
