"""Run the three fixed learning rates sequentially and summarize completed epochs."""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

RATES = ("0.01", "0.1", "0.001")
PROJECT = Path(__file__).resolve().parents[1]
CONFIGS = Path(__file__).with_name("lr_sweep_configs")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def summarize(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, summary = [], {}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = (("train_loss", "Train objective (scheduled beta)"),
              ("val_loss_fixed_beta", "Validation objective (fixed beta)"),
              ("train_full_ce", "Train full reconstruction CE"),
              ("train_kl_per_token", "Train KL per valid codon"),
              ("masked_mutable_ce", "Validation masked CE"))
    for label in RATES:
        run = output / ("lr_" + label)
        metrics = read_rows(run / "metrics.jsonl")
        status_path = output / ("status_" + label + ".json")
        status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "not_started"}
        summary[label] = {**status, "recorded_epochs": len(metrics)}
        if metrics:
            best = min(metrics, key=lambda m: m["val_loss_fixed_beta"])
            summary[label].update(best_val_epoch=best["epoch"], best_val_loss=best["val_loss_fixed_beta"])
        for row in metrics:
            rows.append({"learning_rate_setting": label, "run_status": status["status"], **row})
        for axis, (key, title) in zip(axes.flat, panels):
            axis.plot([r["epoch"] for r in metrics], [r[key] for r in metrics], marker="o", label=label)
            axis.set(title=title, xlabel="Epoch", ylabel="Loss (nats)")
        steps = read_rows(run / "steps.jsonl")
        axes.flat[5].plot([r["step"] for r in steps], [r["loss"] for r in steps], label=label)
    axes.flat[5].set(title="Train loss (logging window mean)", xlabel="Optimizer step", ylabel="Loss (nats)")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(title="Fixed learning rate")
        axis.set_yscale("symlog", linthresh=1.0)
    fig.suptitle("VMAE learning-rate comparison — partial/failed runs are retained")
    fig.tight_layout()
    fig.savefig(output / "loss_comparison.png", dpi=160)
    fig.savefig(output / "loss_comparison.svg")
    plt.close(fig)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["learning_rate_setting", "run_status", "epoch"]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/vmae_lr_sweep_5ep")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.summarize:
        if not output.is_dir():
            raise FileNotFoundError(output)
        summarize(output)
        return
    import matplotlib  # Fail before training if the plotting dependency is missing.
    from .train import load_data, sha256

    configurations = {label: json.loads((CONFIGS / ("lr_" + label + ".json")).read_text()) for label in RATES}
    reference = None
    for label, config in configurations.items():
        settings = config["training"].copy()
        if (settings.pop("learning_rate") != float(label) or settings["epochs"] != 5
                or settings["lr_schedule"] != "constant" or settings["patience"] != 0):
            raise ValueError("Sweep requires the declared fixed learning rates, 5 epochs and no early stopping")
        comparable = {"model": config["model"], "training": settings}
        if reference is not None and comparable != reference:
            raise ValueError("Sweep configurations differ beyond learning rate")
        reference = comparable
    manifest = {"configs": configurations, "dataset_manifest_sha256": sha256(PROJECT / "dataset/manifest.json")}
    if args.resume:
        if json.loads((output / "sweep_manifest.json").read_text()) != manifest:
            raise ValueError("Sweep data/config mismatch; choose a new output directory")
    elif output.exists():
        raise FileExistsError("Sweep output exists; use --resume or a new --output")
    splits, _ = load_data(PROJECT / "dataset", allow_development=True)
    print(json.dumps({"verified_split_sizes": {k: len(v) for k, v in splits.items()}}), flush=True)
    del splits
    output.mkdir(parents=True, exist_ok=True)
    (output / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    failed = False
    for label in RATES:
        run = output / ("lr_" + label)
        status_path = output / ("status_" + label + ".json")
        if args.resume and status_path.exists() and json.loads(status_path.read_text())["status"] == "completed":
            continue
        command = [sys.executable, "-u", "-m", "latent_decoder.train", "--data", str(PROJECT / "dataset"),
                   "--allow-development-split", "--config", str(CONFIGS / ("lr_" + label + ".json")),
                   "--output", str(run)]
        if args.resume and (run / "last.pt").is_file():
            command += ["--resume", str(run / "last.pt")]
        status = {"status": "running", "learning_rate": float(label), "command": command}
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        with (output / ("lr_" + label + ".log")).open("a", encoding="utf-8") as log:
            log.write("\nAttempt " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            log.flush()
            process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace")
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                status.update(status="interrupted")
                status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
                raise
        epochs = [row["epoch"] for row in read_rows(run / "metrics.jsonl")]
        complete = code == 0 and epochs == list(range(1, 6))
        status.update(status="completed" if complete else "failed", exit_code=code)
        status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
        failed |= not complete
        summarize(output)
    if failed:
        raise SystemExit("Some runs failed; inspect status files and logs. Completed runs are preserved.")


if __name__ == "__main__":
    main()
