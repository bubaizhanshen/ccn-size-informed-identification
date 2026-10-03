"""Matched LSTM/Transformer high-CCN comparison on audited SGP histories."""

from __future__ import annotations
from pathlib import Path
import os
from runtime import ROOT
from runtime import WORK

os.umask(63)
WORK.mkdir(parents=True, exist_ok=True, mode=448)
for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "MPLCONFIGDIR"):
    os.environ[key] = str(WORK)
for key in ("TF_CPP_MIN_LOG_LEVEL", "TF_ENABLE_ONEDNN_OPTS"):
    os.environ[key] = {"TF_CPP_MIN_LOG_LEVEL": "3", "TF_ENABLE_ONEDNN_OPTS": "0"}[key]
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import gc
import hashlib
import json
import numpy as np
import pandas as pd
import tensorflow as tf

OUT = ROOT / "results/model_benchmark"
SEEDS = (270927, 270928)
REPRESENTATIONS = ("N", "N82", "N_plus_f82", "full24")
FAMILIES = ("lstm", "transformer")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class PositionEmbedding(tf.keras.layers.Layer):

    def __init__(self, steps: int, width: int):
        super().__init__()
        self.steps = steps
        self.width = width

    def build(self, input_shape):
        self.position = self.add_weight(
            name="position",
            shape=(1, self.steps, self.width),
            initializer=tf.keras.initializers.RandomNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs):
        return inputs + self.position


def representation(
    d: dict[str, np.ndarray], name: str, train: np.ndarray
) -> np.ndarray:
    total = d["x_N"][:, :, None]
    count82 = d["x_N82"][:, :, None]
    fraction82 = d["x_f82"][:, :, None]
    if name == "N":
        x = total
    elif name == "N82":
        x = count82
    elif name == "N_plus_f82":
        x = np.concatenate((total, fraction82), axis=2)
    elif name == "full24":
        x = np.concatenate((total, d["x_fraction24"]), axis=2)
    else:
        raise ValueError(name)
    if not np.isfinite(x).all():
        raise ValueError("nonfinite neural PNSD representation")
    mean = x[train].mean(axis=(0, 1))
    sd = x[train].std(axis=(0, 1))
    sd = np.where(sd > 1e-07, sd, 1.0)
    return ((x - mean[None, None, :]) / sd[None, None, :]).astype("float32")


def build_model(kind: str, steps: int, variables: int, seed: int) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    sequence = tf.keras.Input(shape=(steps, variables), name="pnsd_history")
    calendar = tf.keras.Input(shape=(4,), name="known_calendar")
    if kind == "lstm":
        z = tf.keras.layers.LSTM(32, dropout=0.1)(sequence)
    elif kind == "transformer":
        z = tf.keras.layers.Dense(32)(sequence)
        z = PositionEmbedding(steps, 32)(z)
        att = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16, dropout=0.1)(
            z, z
        )
        z = tf.keras.layers.LayerNormalization()(z + att)
        ff = tf.keras.layers.Dense(64, activation="relu")(z)
        ff = tf.keras.layers.Dense(32)(ff)
        z = tf.keras.layers.LayerNormalization()(z + ff)
        z = tf.keras.layers.GlobalAveragePooling1D()(z)
    else:
        raise ValueError(kind)
    z = tf.keras.layers.Concatenate()((z, calendar))
    z = tf.keras.layers.Dense(16, activation="relu")(z)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(z)
    model = tf.keras.Model((sequence, calendar), output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
    )
    return model


def main() -> None:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    dataset_path = OUT / "common_cohort.npz"
    manifest = json.loads((OUT / "common_cohort_manifest.json").read_text())
    if digest(dataset_path) != manifest["common_cohort_sha256"]:
        raise ValueError("common cohort hash mismatch")
    with np.load(dataset_path, allow_pickle=False) as file:
        d = {key: file[key] for key in file.files}
    train, val, cal, test = (
        d[key].astype(bool) for key in ("train", "val", "cal", "test")
    )
    y = d["y"].astype("float32")
    calendar = d["calendar"].astype("float32")
    runs = OUT / "neural_runs"
    runs.mkdir(parents=True, exist_ok=True, mode=448)
    rows = []
    predictions = []
    for name in REPRESENTATIONS:
        x = representation(d, name, train)
        for kind in FAMILIES:
            for seed in SEEDS:
                label = f"{kind}_{name}_seed{seed}"
                record = runs / f"{label}.json"
                output = runs / f"{label}.csv.gz"
                if record.exists() and output.exists():
                    result = json.loads(record.read_text())
                    if result.get("dataset_sha256") != manifest["common_cohort_sha256"]:
                        raise ValueError(f"stale neural run: {label}")
                    frame = pd.read_csv(output)
                else:
                    model = build_model(kind, x.shape[1], x.shape[2], seed)
                    callback = tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=6,
                        min_delta=0.0001,
                        restore_best_weights=True,
                    )
                    history = model.fit(
                        (x[train], calendar[train]),
                        y[train],
                        validation_data=((x[val], calendar[val]), y[val]),
                        epochs=40,
                        batch_size=64,
                        shuffle=True,
                        verbose=0,
                        callbacks=[callback, tf.keras.callbacks.TerminateOnNaN()],
                    )
                    best = int(np.argmin(history.history["val_loss"]) + 1)
                    pval = model.predict(
                        (x[val], calendar[val]), batch_size=256, verbose=0
                    ).ravel()
                    pcal = model.predict(
                        (x[cal], calendar[cal]), batch_size=256, verbose=0
                    ).ravel()
                    ptest = model.predict(
                        (x[test], calendar[test]), batch_size=256, verbose=0
                    ).ravel()
                    result = {
                        "context": "history6",
                        "representation": name,
                        "model": kind,
                        "seed": seed,
                        "n_train": int(train.sum()),
                        "n_val": int(val.sum()),
                        "n_cal": int(cal.sum()),
                        "n_test": int(test.sum()),
                        "best_epoch": best,
                        "val_brier": float(np.mean((y[val] - pval) ** 2)),
                        "dataset_sha256": manifest["common_cohort_sha256"],
                    }
                    frame = pd.concat(
                        [
                            pd.DataFrame(
                                {
                                    "context": "history6",
                                    "representation": name,
                                    "model": kind,
                                    "seed": seed,
                                    "partition": part,
                                    "block_start": d["block_start"][mask],
                                    "origin": d["origin"][mask],
                                    "event": y[mask].astype(int),
                                    "p_raw": p,
                                }
                            )
                            for part, mask, p in (
                                ("cal", cal, pcal),
                                ("test", test, ptest),
                            )
                        ],
                        ignore_index=True,
                    )
                    frame.to_csv(output, index=False, compression="gzip")
                    record.write_text(json.dumps(result, indent=2), encoding="utf-8")
                    del model, history
                    tf.keras.backend.clear_session()
                    gc.collect()
                rows.append(result)
                predictions.append(frame)
                print(
                    label,
                    "val Brier",
                    round(result["val_brier"], 5),
                    "best epoch",
                    result["best_epoch"],
                    flush=True,
                )
    pd.DataFrame(rows).to_csv(OUT / "neural_fit_summary.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        OUT / "neural_predictions.csv.gz", index=False, compression="gzip"
    )
    (OUT / "neural_manifest.json").write_text(
        json.dumps(
            {
                "dataset_sha256": manifest["common_cohort_sha256"],
                "source_script_sha256": digest(Path(__file__)),
                "models": list(FAMILIES),
                "representations": list(REPRESENTATIONS),
                "seeds": list(SEEDS),
                "context": "history6",
                "architecture": "LSTM(32) or Transformer(Dense32, position embedding, 2x16 attention, one encoder block); static calendar joined before dense16",
                "training": "train-only scaling; Adam .001; batch64; max40 epochs; patience6 chronological validation; CPU",
                "test_use": "prediction and evaluation only; no model or epoch selection",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
