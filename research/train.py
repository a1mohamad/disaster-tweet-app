import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path

# =========================
# Imports from your modules
# =========================
from configs.train_config import TrainConfig
from data_utils.preprocessing import preprocess_df
from data_utils.dataset import build_vocab, save_vocabs, DisasterDataset

from model.model import DisasterTwittsClassifier
from model.embedding import load_embedding

from training.engine import train_one_epoch, evaluate
from training.metrics import build_train_metrics, build_val_metrics
from training.loss import pos_class_weight
from training.callbacks import EarlyStopping, TrainingHistory

from utils.seed import seed_everything


# =========================
# Main training pipeline
# =========================
def main():
    CFG = TrainConfig()
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Seed
    seed_everything(CFG.SEED)

    # 2. Load data
    df = pd.read_csv(CFG.TRAIN_CSV)

    # 3. Preprocess
    df = preprocess_df(df, CFG)

    # 4. Train / validation split
    train_df, val_df = train_test_split(
        df,
        train_size=CFG.TRAIN_TEST_SPLIT,
        stratify=df["target"],
        random_state=CFG.SEED,
        shuffle=True
    )

    # 5. Build vocab (TRAIN ONLY)
    vocab = build_vocab(train_df["final_text"], CFG)
    save_vocabs(vocab, CFG.VOCAB_PATH)

    # 6. Datasets & loaders
    train_ds = DisasterDataset(train_df, vocab, CFG)
    val_ds   = DisasterDataset(val_df, vocab, CFG)

    train_loader = DataLoader(
        train_ds,
        num_workers=CFG.NUM_WORKERS,
        batch_size=CFG.BATCH_SIZE,
        pin_memory=True,
        shuffle=True
    )

    val_loader = DataLoader(
        val_ds,
        num_workers=CFG.NUM_WORKERS,
        batch_size=CFG.BATCH_SIZE,
        pin_memory=True,
        shuffle=False
    )

    # 7. Load GloVe embedding matrix
    embedding_matrix = load_embedding(
        CFG.GLOVE_PATH,
        vocab,
        CFG.EMB_DIM
    )

    # 8. Model
    model = DisasterTwittsClassifier(
        vocab_size=len(vocab),
        emb_dim=CFG.EMB_DIM,
        hidden_dim=CFG.HIDDEN_DIM,
        output_dim=CFG.OUTPUT_DIM,
        num_layers=CFG.NUM_LSTM_LAYERS,
        dropout=CFG.DROPOUT,
        bidirectional=CFG.BIDIRECTIONAL,
        embedding=embedding_matrix,
        freeze_embedding=CFG.FREEZE_EMBEDDING
    ).to(CFG.DEVICE)

    # 9. Loss, optimizer, metrics
    pos_weight = pos_class_weight(train_df, CFG)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LEARNING_RATE)

    train_metrics = build_train_metrics(CFG)
    val_metrics   = build_val_metrics(CFG)

    early_stopping = EarlyStopping(
        patience=CFG.EARLY_STOP_PATIENCE, 
        mode="min",
        min_delta=CFG.EARLY_STOP_MIN_DELTA
        )
    history = TrainingHistory()

    # 10. Training loop
    for epoch in range(CFG.EPOCHS):

        train_loss, train_res = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            train_metrics,
            CFG.DEVICE
        )

        val_loss, val_res = evaluate(
            model,
            val_loader,
            criterion,
            val_metrics,
            CFG.DEVICE
        )

        history.update(
            train_loss,
            val_loss,
            train_res,
            val_res,
            optimizer
        )

        print(
            f"[Epoch {epoch+1}/{CFG.EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val F1: {val_res['f1']:.4f}"
        )

        improved = early_stopping.step(val_loss)
        if improved:
            torch.save(model.state_dict(), CFG.MODEL_PATH)

        if early_stopping.should_stop:
            print("Early stopping triggered.")
            break

    # 11. Save history
    history.save(CFG.HISTORY_PATH)


if __name__ == "__main__":
    main()
