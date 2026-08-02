import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datetime import datetime


MODE = "train"
MODEL_PATH = "model.pth"
EVAL_LOG_FILE = "evaluation_results.txt"

TRAIN_CSV = "train.csv"
TEST_CSV  = "test.csv"

POINTS_PER_DAY = 48
BATCH_SIZE = 64
EPOCHS = 500
LR = 1e-3
USE_CLASS_WEIGHTS = True
DROPOUT = 0.2
SEED = 2025
NUM_WORKERS = 4
PIN_MEMORY = True


RANK_OUT = "ranked_users.csv"
MAP_K_LIST = [100, 200]
SAVE_RANKED_USERS = True

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


def load_table(csv_path: str):
    df = pd.read_csv(csv_path)
    assert "label" in df.columns and "id" in df.columns, "CSV must contain the id and label columns"

    feat_cols = [c for c in df.columns if c not in ("id", "label")]
    X = df[feat_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    ids = df["id"].to_numpy()

    return X, y, ids, feat_cols


def to_binary_labels(y: np.ndarray) -> np.ndarray:
    return (y != 0).astype(np.int64)


def compute_effective_lengths(X: np.ndarray) -> np.ndarray:
    is_nan = np.isnan(X)
    rev = is_nan[:, ::-1]
    first_non_nan_from_end = np.argmax(~rev, axis=1)
    all_nan = np.all(is_nan, axis=1)

    eff_len = X.shape[1] - first_non_nan_from_end
    eff_len[all_nan] = 0

    return eff_len.astype(np.int32)


def choose_fixed_len(eff_len: np.ndarray, points_per_day=48) -> int:
    eff = (eff_len // points_per_day) * points_per_day
    eff[eff == 0] = points_per_day
    return int(eff.min())


class CNNSliceLSTMDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, ids=None, points_per_day=48, fixed_len: int = None):
        self.y = y
        self.ids = ids if ids is not None else np.arange(len(y))
        self.points_per_day = points_per_day

        eff_len = compute_effective_lengths(X)
        if fixed_len is None:
            fixed_len = choose_fixed_len(eff_len, points_per_day)

        self.fixed_len = fixed_len

        start_idx = np.maximum(eff_len - fixed_len, 0)
        rows = []

        for i in range(X.shape[0]):
            s = start_idx[i]
            e = s + fixed_len
            row = X[i, s:e].astype(np.float32, copy=False)

            if row.shape[0] < fixed_len:
                pad = np.zeros(fixed_len - row.shape[0], dtype=np.float32)
                row = np.concatenate([row, pad], axis=0)

            row = np.where(~np.isnan(row), row, np.float32(0.0)).astype(np.float32)
            rows.append(row)

        self.X_fixed = np.stack(rows, axis=0).astype(np.float32)
        assert self.X_fixed.shape[1] % points_per_day == 0
        self.days = self.X_fixed.shape[1] // points_per_day

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        seq = self.X_fixed[idx]

        x_cnn = torch.from_numpy(seq[None, :]).to(torch.float32)
        x_lstm = torch.from_numpy(seq.reshape(self.days, self.points_per_day)).to(torch.float32)

        label = int(self.y[idx])
        user_id = str(self.ids[idx])

        return x_cnn, x_lstm, label, user_id


class SE1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        mid = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        b, c, _ = x.size()
        w = self.pool(x).view(b, c)
        w = self.fc2(F.relu(self.fc1(w), inplace=True)).view(b, c, 1)
        w = torch.sigmoid(w)
        return x * w


class TemporalAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x):
        a = torch.cat(
            [
                x.mean(1, keepdim=True),
                x.amax(1, keepdim=True)
            ],
            dim=1
        )
        a = torch.sigmoid(self.conv(a))
        return x * a


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()

        pad = (kernel_size // 2) * dilation

        self.depth = nn.Conv1d(
            in_ch,
            in_ch,
            kernel_size,
            padding=pad,
            dilation=dilation,
            groups=in_ch,
            bias=False
        )

        self.point = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class MultiScaleBlockTiny(nn.Module):
    def __init__(self, in_ch, out_ch, kernels=(3, 5), dilations=(1, 2), att_kernel=5, drop=0.1):
        super().__init__()

        branches = []
        per = max(1, out_ch // (len(kernels) * len(dilations)))

        for k in kernels:
            for d in dilations:
                branches.append(
                    DepthwiseSeparableConv1d(
                        in_ch,
                        per,
                        k,
                        dilation=d
                    )
                )

        self.branches = nn.ModuleList(branches)
        self.proj = nn.Conv1d(per * len(branches), out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.se = SE1D(out_ch, reduction=8)
        self.ta = TemporalAttention(att_kernel)
        self.drop = nn.Dropout(drop)

        self.res = nn.Identity() if in_ch == out_ch else nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.res_bn = nn.BatchNorm1d(out_ch) if not isinstance(self.res, nn.Identity) else nn.Identity()

    def forward(self, x):
        z = torch.cat([b(x) for b in self.branches], dim=1)
        z = self.proj(z)
        z = self.bn(z)
        z = F.relu(z, inplace=True)
        z = self.se(z)
        z = self.ta(z)
        z = self.drop(z)

        r = self.res_bn(self.res(x))

        return F.relu(z + r, inplace=True)


class SeasonalPositionalEncodingTiny(nn.Module):
    def __init__(self, L, periods=(48, 336)):
        super().__init__()

        t = torch.arange(L, dtype=torch.float32)
        pe = []

        for P in periods:
            pe.append(torch.sin(2 * torch.pi * t / P))
            pe.append(torch.cos(2 * torch.pi * t / P))

        pe = torch.stack(pe, dim=0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        B, _, L = x.shape
        pe = self.pe[:, :L].unsqueeze(0).expand(B, -1, -1)
        return torch.cat([x, pe], dim=1)


class DayWeekHeadTiny(nn.Module):
    def __init__(self, in_ch, day_hid=32, week_out=64, dilations=(1, 2), points_per_day=48, drop=0.1):
        super().__init__()

        self.ppd = points_per_day

        self.day_enc = nn.Sequential(
            nn.Conv1d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, day_hid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

        blocks = []
        for d in dilations:
            blocks += [
                nn.Conv1d(day_hid, day_hid, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm1d(day_hid),
                nn.ReLU(inplace=True)
            ]

        self.tcn = nn.Sequential(*blocks)

        self.week_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(drop),
            nn.Linear(day_hid, week_out)
        )

    def forward(self, xin):
        B, C, L = xin.shape
        D = L // self.ppd

        xd = xin.view(B, C, D, self.ppd)

        days = []
        for d in range(D):
            fd = self.day_enc(xd[:, :, d, :]).squeeze(-1)
            days.append(fd)

        day_seq = torch.stack(days, dim=2)
        z = self.tcn(day_seq)

        return self.week_head(z)


class GateFusion(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.proj = nn.Linear(in_dim, out_dim)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Sigmoid()
        )

    def forward(self, *feats):
        z = torch.cat(feats, dim=1)
        return self.proj(z) * self.gate(z)


class MSACNNPlusTiny(nn.Module):
    def __init__(self, num_classes: int, L: int, points_per_day=48,
                 base_ch=32, blocks=2, width_mult=1.0, dropout=0.15):
        super().__init__()

        self.spe = SeasonalPositionalEncodingTiny(L, periods=(48, 336))
        in_ch = 5

        def C(ch):
            return max(8, int(ch * width_mult))

        stem_ch = C(base_ch)

        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, stem_ch, 7, padding=3, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.ReLU(inplace=True)
        )

        ch = stem_ch
        enc_blocks = []

        for i in range(blocks):
            out_ch = ch if i == 0 else C(ch * 2)

            enc_blocks.append(
                MultiScaleBlockTiny(
                    ch,
                    out_ch,
                    kernels=(3, 5),
                    dilations=(1, 2),
                    att_kernel=5,
                    drop=dropout
                )
            )

            if i > 0:
                ch = out_ch

        self.encoder = nn.Sequential(*enc_blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)

        self.dw_head = DayWeekHeadTiny(
            in_ch=in_ch,
            day_hid=C(32),
            week_out=C(64),
            dilations=(1, 2),
            points_per_day=points_per_day,
            drop=dropout
        )

        fuse_in = ch + C(64)

        self.fuse = GateFusion(fuse_in, C(96))

        self.head = nn.Sequential(
            nn.Linear(C(96), C(96)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(C(96), num_classes)
        )

    def forward(self, x_cnn, x_lstm=None):
        xin = self.spe(x_cnn)

        z = self.stem(xin)
        z = self.encoder(z)
        g = self.gap(z).squeeze(-1)

        t = self.dw_head(xin)

        h = self.fuse(g, t)

        return self.head(h)


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total / 1e6:.3f}M, trainable={trainable / 1e6:.3f}M")


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()

    total_loss, total = 0.0, 0

    for x_cnn, x_lstm, y, _ in tqdm(loader, desc="Train", leave=False):
        x_cnn = x_cnn.to(device, dtype=torch.float32)
        x_lstm = x_lstm.to(device, dtype=torch.float32)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x_cnn, x_lstm)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)

    return total_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss, total = 0.0, 0
    all_trues, all_scores, all_ids = [], [], []

    for x_cnn, x_lstm, y, user_id in tqdm(loader, desc="Eval", leave=False):
        x_cnn = x_cnn.to(device, dtype=torch.float32)
        x_lstm = x_lstm.to(device, dtype=torch.float32)
        y = y.to(device)

        logits = model(x_cnn, x_lstm)
        loss = criterion(logits, y)


        anomaly_score = torch.softmax(logits, dim=1)[:, 1]

        total_loss += loss.item() * y.size(0)
        total += y.size(0)

        all_trues.append(y.cpu().numpy())
        all_scores.append(anomaly_score.cpu().numpy())
        all_ids.extend(list(user_id))

    avg_loss = total_loss / total
    y_true = np.concatenate(all_trues)
    y_score = np.concatenate(all_scores)

    return avg_loss, y_true, y_score, all_ids


def average_precision_at_k(y_true, y_score, k=100):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    k = min(k, len(y_true))

    order = np.argsort(y_score)[::-1]
    top_k = y_true[order][:k]

    precisions = []
    hit_count = 0

    for rank, label in enumerate(top_k, start=1):
        if label == 1:
            hit_count += 1
            precisions.append(hit_count / rank)

    if len(precisions) == 0:
        return 0.0

    return float(np.mean(precisions))


def compute_metrics(y_true, y_score):
    metrics = {}

    try:
        metrics["AUC"] = roc_auc_score(y_true, y_score)
    except ValueError:
        metrics["AUC"] = float("nan")

    for k in MAP_K_LIST:
        metrics[f"MAP@{k}"] = average_precision_at_k(y_true, y_score, k=k)

    return metrics


def print_metrics(metrics):
    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        if np.isnan(v):
            print(f"  {k}: NaN")
        else:
            print(f"  {k}: {v:.4f}")


def log_metrics(
    metrics,
    mode="train",
    epoch=None,
    train_loss=None,
    valid_loss=None,
    test_loss=None,
    best_saved=False,
    filepath=EVAL_LOG_FILE
):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(filepath, "a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Time: {now}\n")
        f.write(f"Mode: {mode}\n")

        if epoch is not None:
            f.write(f"Epoch: {epoch}\n")

        if train_loss is not None:
            f.write(f"Train Loss: {train_loss:.6f}\n")

        if valid_loss is not None:
            f.write(f"Valid Loss: {valid_loss:.6f}\n")

        if test_loss is not None:
            f.write(f"Test Loss: {test_loss:.6f}\n")

        f.write("Evaluation Metrics:\n")
        for k, v in metrics.items():
            if np.isnan(v):
                f.write(f"  {k}: NaN\n")
            else:
                f.write(f"  {k}: {v:.4f}\n")

        if best_saved:
            f.write(f"Best Model Saved To: {MODEL_PATH}\n")

        f.write("\n")


def save_ranked_users(ids, y_true, y_score, out_path=RANK_OUT):
    result_df = pd.DataFrame({
        "id": ids,
        "true_binary_label": y_true,
        "anomaly_score": y_score
    })

    result_df = result_df.sort_values("anomaly_score", ascending=False)
    result_df["rank"] = np.arange(1, len(result_df) + 1)

    result_df = result_df[
        [
            "rank",
            "id",
            "true_binary_label",
            "anomaly_score"
        ]
    ]

    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"Ranked users saved to: {out_path}")


def save_checkpoint(model, fixed_len, path=MODEL_PATH):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "fixed_len": fixed_len,
        "num_classes": 2,
        "points_per_day": POINTS_PER_DAY,
        "base_ch": 32,
        "blocks": 2,
        "width_mult": 1.0,
        "dropout": 0.15,
    }

    torch.save(ckpt, path)
    print(f"Model saved to: {path}")


def load_checkpoint(path=MODEL_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found: {path}. Please set MODE='train' first to train and save the model."
        )

    ckpt = torch.load(path, map_location=device)

    model = MSACNNPlusTiny(
        num_classes=ckpt.get("num_classes", 2),
        L=ckpt["fixed_len"],
        points_per_day=ckpt.get("points_per_day", POINTS_PER_DAY),
        base_ch=ckpt.get("base_ch", 32),
        blocks=ckpt.get("blocks", 2),
        width_mult=ckpt.get("width_mult", 1.0),
        dropout=ckpt.get("dropout", 0.15),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    print(f"Model loaded from: {path}")

    return model, ckpt["fixed_len"]


def build_loaders(fixed_len=None):
    Xtr_raw, ytr, ids_tr, feat_cols = load_table(TRAIN_CSV)
    Xte_raw, yte, ids_te, _ = load_table(TEST_CSV)


    ytr = to_binary_labels(ytr)
    yte = to_binary_labels(yte)

    if fixed_len is None:
        eff_tr = compute_effective_lengths(Xtr_raw)
        fixed_len = choose_fixed_len(eff_tr, POINTS_PER_DAY)

    print(f"Using fixed sequence length L={fixed_len} (approximately {fixed_len // POINTS_PER_DAY} days × {POINTS_PER_DAY} points/day)")
    print(f"Train label distribution: {dict(zip(*np.unique(ytr, return_counts=True)))}")
    print(f"Test label distribution:  {dict(zip(*np.unique(yte, return_counts=True)))}")

    train_ds = CNNSliceLSTMDataset(
        Xtr_raw,
        ytr,
        ids=ids_tr,
        points_per_day=POINTS_PER_DAY,
        fixed_len=fixed_len
    )

    test_ds = CNNSliceLSTMDataset(
        Xte_raw,
        yte,
        ids=ids_te,
        points_per_day=POINTS_PER_DAY,
        fixed_len=fixed_len
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False
    )

    return train_loader, test_loader, fixed_len, ytr


def train_mode():
    train_loader, test_loader, fixed_len, ytr = build_loaders(fixed_len=None)

    model = MSACNNPlusTiny(
        num_classes=2,
        L=fixed_len,
        points_per_day=POINTS_PER_DAY,
        base_ch=32,
        blocks=2,
        width_mult=1.0,
        dropout=0.15
    ).to(device)

    count_params(model)

    if USE_CLASS_WEIGHTS:
        classes = np.array([0, 1])
        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=ytr
        )
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print("Class weights:", weights)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_auc = -1.0

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")

        tr_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler
        )

        te_loss, y_true, y_score, ids = evaluate(
            model,
            test_loader,
            criterion
        )

        metrics = compute_metrics(y_true, y_score)

        print(f"  train loss {tr_loss:.4f}")
        print(f"  valid loss {te_loss:.4f}")
        print_metrics(metrics)

        current_auc = metrics["AUC"]
        saved_best = False


        if not np.isnan(current_auc) and current_auc > best_auc:
            best_auc = current_auc
            save_checkpoint(model, fixed_len, MODEL_PATH)
            print("  (saved best model)")
            saved_best = True


        log_metrics(
            metrics,
            mode="train_epoch_eval",
            epoch=epoch,
            train_loss=tr_loss,
            valid_loss=te_loss,
            best_saved=saved_best,
            filepath=EVAL_LOG_FILE
        )

    print("\nReloading best model and evaluating...")

    best_model, fixed_len = load_checkpoint(MODEL_PATH)

    _, test_loader, _, _ = build_loaders(fixed_len=fixed_len)

    criterion = nn.CrossEntropyLoss()

    te_loss, y_true, y_score, ids = evaluate(
        best_model,
        test_loader,
        criterion
    )

    metrics = compute_metrics(y_true, y_score)

    print(f"\nBest model -> test loss {te_loss:.4f}")
    print_metrics(metrics)


    log_metrics(
        metrics,
        mode="best_model_final_eval",
        test_loss=te_loss,
        filepath=EVAL_LOG_FILE
    )

    if SAVE_RANKED_USERS:
        save_ranked_users(ids, y_true, y_score, RANK_OUT)


def test_mode():
    model, fixed_len = load_checkpoint(MODEL_PATH)

    _, test_loader, _, _ = build_loaders(fixed_len=fixed_len)

    criterion = nn.CrossEntropyLoss()

    te_loss, y_true, y_score, ids = evaluate(
        model,
        test_loader,
        criterion
    )

    metrics = compute_metrics(y_true, y_score)

    print(f"\nTest result -> loss {te_loss:.4f}")
    print_metrics(metrics)


    log_metrics(
        metrics,
        mode="test_eval",
        test_loss=te_loss,
        filepath=EVAL_LOG_FILE
    )

    if SAVE_RANKED_USERS:
        save_ranked_users(ids, y_true, y_score, RANK_OUT)


def main():
    if MODE.lower() == "train":
        train_mode()
    elif MODE.lower() == "test":
        test_mode()
    else:
        raise ValueError("MODE must be set to either 'train' or 'test'")


if __name__ == "__main__":
    main()
