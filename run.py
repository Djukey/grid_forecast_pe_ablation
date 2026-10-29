"""Positional-encoding ablation for 24-hour electricity-demand forecasting on one GPU.

    python run.py                      # full study, regenerates everything in results/
    python run.py --config config_smoke.yaml --out results_smoke   # fast pipeline check
    python run.py --figures-only       # redraw figures from the committed results/*.csv (no GPU needed)

Outputs (in the output folder):
    runs.csv              raw measurements, one row per (repeat, setting, seed)
    summary.csv / .md     per-setting statistics and the noise check
    checks.json           baseline error, determinism check, data checksum
    example_forecast.csv  one test day: actual and every forecast
    environment.json      full hardware / software / workload record (logged automatically)
    hardware_footnote.txt one-line footnote used under every table and figure
    requirements_used.txt exact library versions of this run
    mae_by_setting.png, example_forecast.png   figures drawn from the CSV files above
"""
import argparse, hashlib, json, math, os, platform, random, subprocess, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, yaml
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COLORS = {"none": "#2a78d6", "sinusoidal": "#eb6834", "learned": "#1baf7a"}   # one fixed colour per setting
T95 = {1: float("nan"), 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}  # two-sided t, n-1 df


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------- 1. Data: synthetic hourly demand for one rural distribution area ----------
def make_demand(days, seed):
    rng = np.random.default_rng(seed)
    h = np.arange(days * 24); hour, day = h % 24, h // 24
    daily = 1.0 + 0.35 * np.exp(-((hour - 8) ** 2) / 6) + 0.55 * np.exp(-((hour - 19) ** 2) / 8)  # morning + evening peaks
    weekly = np.where(day % 7 >= 5, 0.85, 1.0)                                                     # weekends lower
    temp = 10 - 12 * np.cos(2 * np.pi * day / 365) + rng.normal(0, 3, len(h))                      # deg C, cold winters
    heating, cooling = np.clip(15 - temp, 0, None) * 0.03, np.clip(temp - 22, 0, None) * 0.04
    demand_mw = 5.0 * daily * weekly * (1 + heating + cooling) + rng.normal(0, 0.25, len(h))
    return pd.DataFrame({"hour_index": h, "demand_mw": demand_mw.round(3)})


def windows(series, lookback, horizon):
    X = np.lib.stride_tricks.sliding_window_view(series, lookback + horizon)
    return torch.tensor(X[:, :lookback], dtype=torch.float32), torch.tensor(X[:, lookback:], dtype=torch.float32)


# ---------- 2. Model: tiny transformer; ONLY the positional encoding changes ----------
class Forecaster(nn.Module):
    def __init__(self, pe, lookback, horizon, d_model, layers, heads):
        super().__init__()
        self.pe = pe
        self.inp = nn.Linear(1, d_model)
        if pe == "learned":
            self.pos = nn.Parameter(torch.randn(lookback, d_model) * 0.02)
        elif pe == "sinusoidal":
            pos, i = torch.arange(lookback)[:, None], torch.arange(0, d_model, 2)
            table = torch.zeros(lookback, d_model)
            table[:, 0::2] = torch.sin(pos / 10000 ** (i / d_model)); table[:, 1::2] = torch.cos(pos / 10000 ** (i / d_model))
            self.register_buffer("pos", table)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, horizon)

    def forward(self, x):                        # x: (batch, lookback) normalized demand
        h = self.inp(x[..., None])
        if self.pe != "none": h = h + self.pos   # "none": the model does not know which hour is which
        return self.head(self.enc(h).mean(1))    # average over hours, then predict 24 values


# ---------- 3. One training run = one (setting, seed) ----------
def run_one(setting, seed, cfg, data):
    set_seed(seed)
    m = cfg["model"]; model = Forecaster(setting, m["lookback"], m["horizon"], m["d_model"], m["layers"], m["heads"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), cfg["train"]["lr"])
    Xtr, Ytr, Xte, Yte, mu, sd = data
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(cfg["train"]["epochs"]):
        for idx in torch.randperm(len(Xtr)).split(cfg["train"]["batch_size"]):
            loss = nn.functional.mse_loss(model(Xtr[idx].to(DEVICE)), Ytr[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
    if DEVICE == "cuda": torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    model.eval(); eb, lat = cfg["train"]["eval_batch_size"], cfg["latency"]
    with torch.no_grad():
        pred = torch.cat([model(Xte[i:i + eb].to(DEVICE)).cpu() for i in range(0, len(Xte), eb)])
        mae_mw = float((pred - Yte).abs().mean() * sd)                       # back to megawatts
        x1, times = Xte[:lat["batch_size"]].to(DEVICE), []                   # latency: one forecast at a time
        for k in range(lat["warmup"] + lat["trials"]):
            if DEVICE == "cuda": torch.cuda.synchronize()
            t = time.perf_counter(); model(x1)
            if DEVICE == "cuda": torch.cuda.synchronize()
            if k >= lat["warmup"]: times.append((time.perf_counter() - t) * 1000)
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if DEVICE == "cuda" else float("nan")
    return {"setting": setting, "seed": seed, "test_mae_mw": round(mae_mw, 4), "train_s": round(train_s, 2),
            "p50_ms": round(float(np.percentile(times, 50)), 3), "p95_ms": round(float(np.percentile(times, 95)), 3),
            "peak_mem_mb": round(peak_mb, 1), "params": sum(p.numel() for p in model.parameters() if p.requires_grad)}, pred


# ---------- 4. Automatic hardware / runtime record ----------
def smi(field):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else "unavailable"
    except Exception:
        return "unavailable"


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"): return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def hardware_record(cfg):
    rec = {"python": platform.python_version(), "os": platform.platform(), "cpu": cpu_model(), "cpu_threads": torch.get_num_threads(),
           "torch": torch.__version__, "numpy": np.__version__, "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
           "pyyaml": yaml.__version__, "device": DEVICE,
           "precision": {"weights_and_activations": "float32", "autocast": False,
                         "float32_matmul_precision": torch.get_float32_matmul_precision(),
                         "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32)},
           "workload": {"input_sequence_hours": cfg["model"]["lookback"], "output_horizon_hours": cfg["model"]["horizon"],
                        "train_batch_size": cfg["train"]["batch_size"], "eval_batch_size": cfg["train"]["eval_batch_size"],
                        "latency_batch_size": cfg["latency"]["batch_size"], "epochs": cfg["train"]["epochs"],
                        "seeds": cfg["ablation"]["seeds"], "repeats": cfg.get("repeats", 1)}}
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        rec.update({"gpu": p.name, "vram_gb": round(p.total_memory / 2 ** 30, 2), "compute_capability": f"{p.major}.{p.minor}",
                    "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "driver": smi("driver_version")})
    return rec


def footnote(rec, wall_s):
    w = rec["workload"]
    hw = (f"{rec['gpu']} ({rec['vram_gb']} GB VRAM), driver {rec['driver']}, CUDA runtime {rec['cuda_runtime']}"
          if DEVICE == "cuda" else f"CPU only ({rec['cpu']}, {rec['cpu_threads']} threads)")
    return (f"Hardware: {hw}; PyTorch {rec['torch']}, Python {rec['python']}; precision float32; train batch {w['train_batch_size']}, "
            f"eval batch {w['eval_batch_size']}, latency batch {w['latency_batch_size']}; input {w['input_sequence_hours']} h -> "
            f"{w['output_horizon_hours']} h; {w['epochs']} epochs; seeds {w['seeds']}; study repeated {w['repeats']}x; "
            f"total wall-clock {wall_s / 60:.1f} min.")


# ---------- 5. Figures: drawn ONLY from the CSV / JSON files written above ----------
def render_figures(out):
    runs = pd.read_csv(out / "runs.csv"); summ = pd.read_csv(out / "summary.csv", index_col=0)
    checks = json.load(open(out / "checks.json")); note = open(out / "hardware_footnote.txt").read().strip()
    fc = pd.read_csv(out / "example_forecast.csv"); names = list(summ.index); base = checks["seasonal_naive_mae_mw"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    wrap = lambda s: "\n".join(__import__("textwrap").wrap(s, 105))

    fig, ax = plt.subplots(figsize=(7, 4.8)); r0 = runs[runs.repeat == 0]
    for k, s in enumerate(names):
        ax.bar(k, summ.loc[s, "mae_mean"], width=0.5, color=COLORS.get(s, "#888"), alpha=0.35)
        ax.vlines(k, summ.loc[s, "ci95_low"], summ.loc[s, "ci95_high"], color="#222", lw=1.6)
        ax.scatter([k] * (r0.setting == s).sum(), r0[r0.setting == s].test_mae_mw, s=34, color=COLORS.get(s, "#888"), edgecolor="white", linewidth=1.2, zorder=3)
    ax.axhline(base, color="#555", ls="--", lw=1.2)
    ax.set_xlim(-0.6, len(names) + 0.6)
    ax.text(len(names) - 0.42, base, f"same hour last week\n{base:.3f} MW", ha="left", va="center", color="#333", fontsize=9)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names); ax.set_ylabel("Test MAE (MW), lower is better")
    ax.set_title("Positional encoding vs. forecast error\nbar = mean of seeds, dots = each seed, line = 95% CI", fontsize=10, loc="left")
    fig.text(0.01, 0.01, wrap(note), fontsize=6.5, color="#555", va="bottom"); fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "mae_by_setting.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.4)); ax.plot(fc.hour_ahead, fc.actual, color="#111", lw=2, label="actual")
    ax.plot(fc.hour_ahead, fc.seasonal_naive, color="#777", ls="--", lw=1.4, label="same hour last week")
    for s in names: ax.plot(fc.hour_ahead, fc[s], color=COLORS.get(s, "#888"), lw=1.6, label=s)
    ax.set_xlabel("hours ahead"); ax.set_ylabel("Demand (MW)"); ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.text(0.01, 0.01, wrap(note), fontsize=6.5, color="#555", va="bottom"); fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out / "example_forecast.png", dpi=130); plt.close(fig)


# ---------- 6. Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml")); ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--figures-only", action="store_true", help="redraw figures from existing CSV files in --out")
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        render_figures(out); print("figures redrawn from", out); return

    cfg = yaml.safe_load(open(args.config)); (ROOT / "data").mkdir(exist_ok=True); T0 = time.perf_counter()
    df = make_demand(cfg["data"]["days"], cfg["data"]["data_seed"]); df.to_csv(ROOT / "data" / "demand.csv", index=False)
    data_sha = hashlib.sha256(open(ROOT / "data" / "demand.csv", "rb").read()).hexdigest()
    split = int(len(df) * (1 - cfg["data"]["test_share"])); lb, hz = cfg["model"]["lookback"], cfg["model"]["horizon"]
    train_s, test_s = df.demand_mw.values[:split], df.demand_mw.values[split - lb:]
    mu, sd = train_s.mean(), train_s.std()                                   # normalize with TRAIN statistics only
    Xtr, Ytr = windows((train_s - mu) / sd, lb, hz); Xtr, Ytr = Xtr[::cfg["data"]["train_stride"]], Ytr[::cfg["data"]["train_stride"]]
    Xte, Yte = windows((test_s - mu) / sd, lb, hz); data = (Xtr, Ytr, Xte, Yte, mu, sd)
    base_mae = float((Xte[:, :hz] - Yte).abs().mean() * sd)                  # same hour last week = first 24 h of the 168 h window
    rec = hardware_record(cfg)
    print(f"device {DEVICE} ({rec.get('gpu', rec['cpu'])}) | {len(df)} hours | train windows {len(Xtr)} | test windows {len(Xte)}")
    print(f"baseline (same hour last week): MAE {base_mae:.3f} MW")

    rows, example, reps = [], {}, cfg.get("repeats", 1)
    for rep in range(reps):
        for setting in cfg["ablation"]["settings"]:
            for seed in cfg["ablation"]["seeds"]:
                r, pred = run_one(setting, seed, cfg, data); r = {"repeat": rep, **r}; rows.append(r)
                if rep == 0 and seed == cfg["ablation"]["seeds"][0]: example[setting] = pred[-1].numpy() * sd + mu
                print(f"rep {rep} {setting:10s} seed {seed}: MAE {r['test_mae_mw']:.3f} MW | train {r['train_s']} s | "
                      f"p50 {r['p50_ms']} ms p95 {r['p95_ms']} ms | peak {r['peak_mem_mb']} MB")
    wall = time.perf_counter() - T0
    runs = pd.DataFrame(rows); runs.to_csv(out / "runs.csv", index=False)

    # Accuracy: seeds are the independent unit, so statistics use repeat 0 only. Repeats re-run the SAME seeds,
    # so they check determinism and timing spread; pooling them would fake extra sample size.
    r0 = runs[runs.repeat == 0]; g = r0.groupby("setting", sort=False); n = len(cfg["ablation"]["seeds"])
    half = T95.get(n, 1.96) * g.test_mae_mw.std() / math.sqrt(n)
    # Timing: one value per (setting, repeat) = median over seeds; then mean and spread across repeats.
    per_rep = runs.groupby(["setting", "repeat"], sort=False).agg(p50=("p50_ms", "median"), p95=("p95_ms", "median"), train=("train_s", "mean")).reset_index()
    gt = per_rep.groupby("setting", sort=False)
    summary = pd.DataFrame({"mae_mean": g.test_mae_mw.mean(), "mae_std": g.test_mae_mw.std(),
                            "ci95_low": g.test_mae_mw.mean() - half, "ci95_high": g.test_mae_mw.mean() + half,
                            "mae_min": g.test_mae_mw.min(), "mae_max": g.test_mae_mw.max(),
                            "p50_ms_mean": gt.p50.mean(), "p50_ms_sd": gt.p50.std(), "p95_ms_mean": gt.p95.mean(), "p95_ms_sd": gt.p95.std(),
                            "train_s_mean": gt.train.mean(), "train_s_sd": gt.train.std(),
                            "peak_mem_mb": runs.groupby("setting", sort=False).peak_mem_mb.max(), "params": g.params.first()}).round(4)
    summary.to_csv(out / "summary.csv")

    mae_by_key = runs.pivot_table(index=["setting", "seed"], columns="repeat", values="test_mae_mw")
    det = float((mae_by_key.sub(mae_by_key[0], axis=0)).abs().max().max()) if reps > 1 else None
    checks = {"seasonal_naive_mae_mw": round(base_mae, 4), "mae_max_abs_diff_across_repeats": det, "repeats": reps,
              "seeds_per_setting": n, "data_sha256": data_sha, "wall_clock_s": round(wall, 1)}
    json.dump(checks, open(out / "checks.json", "w"), indent=1)

    names, verdicts = list(summary.index), []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = summary.loc[names[i]], summary.loc[names[j]]
            sep = a.ci95_high < b.ci95_low or b.ci95_high < a.ci95_low
            verdicts.append(f"- {names[i]} vs {names[j]}: difference {abs(a.mae_mean - b.mae_mean):.3f} MW -> "
                            + ("EXCEEDS run-to-run noise (95% CIs do not overlap)" if sep else "WITHIN run-to-run noise (95% CIs overlap): no difference can be claimed"))
    for s in names:
        a = summary.loc[s]
        v = ("the model is WORSE than the baseline" if base_mae < a.ci95_low else "the model is BETTER than the baseline" if base_mae > a.ci95_high
             else "the baseline lies INSIDE the model's CI: no difference can be claimed")
        verdicts.append(f"- {s} vs 'same hour last week' ({base_mae:.3f} MW): {v}")
    note = footnote(rec, wall); open(out / "hardware_footnote.txt", "w").write(note + "\n")
    with open(out / "summary.md", "w") as f:
        f.write(f"## Test MAE (MW) by positional-encoding setting ({n} seeds each)\n\n```\n{summary.to_string()}\n```\n\n"
                f"## Noise check (95% CI overlap rule)\n\n" + "\n".join(verdicts) + f"\n\n## Reproducibility check\n\n"
                f"Max |MAE difference| between repeats with identical seeds: {det}\n\n{note}\n")
    print("\n" + summary.to_string() + "\n\nNOISE CHECK\n" + "\n".join(verdicts) + f"\n\nDeterminism (max MAE diff across {reps} repeats): {det}\n" + note)

    ex = pd.DataFrame({"hour_ahead": range(1, hz + 1), "actual": Yte[-1].numpy() * sd + mu, "seasonal_naive": Xte[-1, :hz].numpy() * sd + mu})
    for s in names: ex[s] = example[s]
    ex.round(4).to_csv(out / "example_forecast.csv", index=False)
    rec["data_sha256"] = data_sha; rec["wall_clock_s"] = round(wall, 1); rec["config"] = cfg
    json.dump(rec, open(out / "environment.json", "w"), indent=1)
    open(out / "requirements_used.txt", "w").write(
        f"torch=={torch.__version__.split('+')[0]}\nnumpy=={np.__version__}\npandas=={pd.__version__}\nmatplotlib=={matplotlib.__version__}\npyyaml=={yaml.__version__}\n")
    render_figures(out); print(f"\nsaved to {out}/")


if __name__ == "__main__":
    main()
