"""Train/evaluate the prepared decoder on the server; never runs on import."""
import argparse
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import (AA, AA_IDS, CODONS, PAD, SYNONYMS, CodonVMAE, collate_records,
                    corrupt, encode_cds, mutable_positions, synonymous_logits, vae_loss)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_data(directory, allow_development=False):
    directory = Path(directory)
    if (directory / "manifest.json").exists():
        from .prepare_combined import verify
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if not manifest.get("source_verified"):
            raise ValueError("Source is not verified; inspect manifest.json")
        if not manifest.get("homology_certified") and not allow_development:
            raise ValueError("Run protein clustering first, or explicitly use --allow-development-split")
        return verify(directory)
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if not audit.get("source_verified"):
        raise ValueError("Source is not verified; inspect audit.json")
    if not audit.get("homology_certified") and not allow_development:
        raise ValueError("Run the documented protein clustering first, or explicitly use "
                         "--allow-development-split for a pilot with no homology-generalization claim")
    if audit.get("corpus_sha256") != sha256(directory / "corpus.jsonl"):
        raise ValueError("Corpus checksum does not match audit.json")
    splits = {name: [] for name in ("train", "val", "test")}
    seen_cds, seen_protein, seen_gene, seen_cluster = {}, {}, {}, {}
    with (directory / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            split = record["split"]
            if split not in splits:
                raise ValueError("Unknown split: " + split)
            ids = encode_cds(record["cds_dna"])
            if record["n_codons"] != len(ids):
                raise ValueError("Length metadata mismatch")
            protein = "".join(AA[i] for i in ids[:-1])
            if record["protein_sequence"].rstrip("*") != protein:
                raise ValueError("Protein annotation mismatch")
            if record["cds_dna"] in seen_cds:
                raise ValueError("Duplicate CDS in prepared corpus")
            seen_cds[record["cds_dna"]] = split
            for table, keys in ((seen_protein, [protein]), (seen_gene, record["gene_ids"]),
                                (seen_cluster, [record["cluster_id"]])):
                for key in keys:
                    if key in table and table[key] != split:
                        raise ValueError("Cross-split identity leakage: " + key)
                    table[key] = split
            splits[split].append(record)
    if any(not records for records in splits.values()):
        raise ValueError("All three splits must be nonempty")
    return splits, audit


def batches(records, size, shuffle=False):
    indices = list(range(len(records)))
    if shuffle:
        random.shuffle(indices)
    # Small shuffled length buckets reduce padding without fixed short-to-long curricula.
    ordered_batches = []
    for start in range(0, len(indices), size * 32):
        bucket = sorted(indices[start:start + size * 32], key=lambda i: records[i]["n_codons"])
        ordered_batches.extend(bucket[j:j + size] for j in range(0, len(bucket), size))
    if shuffle:
        random.shuffle(ordered_batches)
    return DataLoader(records, batch_sampler=ordered_batches, collate_fn=collate_records,
                      num_workers=0)


def codon_frequency(train_records):
    counts = torch.ones(64)  # Training-only Laplace smoothing within each synonymous family.
    for record in train_records:
        ids = encode_cds(record["cds_dna"])
        counts += torch.bincount(torch.tensor(ids), minlength=64)
    return counts


def learning_rate_factor(step, schedule, warmup, total_steps, extension=None):
    if extension and step >= extension["start_step"]:
        progress = min(1., (step - extension["start_step"]) /
                       (extension["end_step"] - extension["start_step"]))
        return extension["start_factor"] * 0.5 * (1 + math.cos(math.pi * progress))
    if schedule == "constant":
        return 1.0
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1., (step - warmup) / max(1, total_steps - warmup))
    return 0.5 * (1 + math.cos(math.pi * progress))


def preflight(model, records, device, batch_size, mask_probability):
    """Check the longest training batch in FP32 without updating weights or RNG state."""
    gpu_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    longest = sorted(records, key=lambda r: r["n_codons"], reverse=True)[:batch_size]
    was_training = model.training
    try:
        with torch.random.fork_rng(devices=gpu_devices):
            model.train()
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            tokens = collate_records(longest).to(device)
            corrupted, selected = corrupt(tokens, mask_probability)
            logits, mu, logvar = model(corrupted, tokens)
            loss, _ = vae_loss(logits, tokens, mu, logvar, selected, beta=0.001)
            if not torch.isfinite(loss):
                raise FloatingPointError("Preflight loss is nonfinite")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            if norm == 0:
                raise FloatingPointError("Preflight gradients are all zero")
            return {"event": "preflight_pass", "batch_size": len(longest),
                    "max_codons": tokens.shape[1], "loss": loss.item(), "precision": "float32",
                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_training)


@torch.no_grad()
def evaluate(model, records, counts, device, batch_size, mask_probability=0.15):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(173)
    totals = {k: 0.0 for k in ("sites", "masked_sites", "clean_ce", "clean_correct",
                               "masked_ce", "masked_correct", "frequency_ce", "frequency_correct",
                               "zero_latent_ce", "shuffled_latent_ce", "invalid", "valid_tokens",
                               "kl", "masked_full_ce", "masked_kl", "sequences", "exact_sequences")}
    latent_sum = latent_sq = None
    latent_count = 0
    counts = counts.to(device)
    for tokens in batches(records, batch_size):
        tokens = tokens.to(device)
        valid = tokens < PAD
        mutable = mutable_positions(tokens)
        logits, mu, logvar = model(tokens, tokens, sample=False)
        logits = synonymous_logits(logits, tokens)
        targets = tokens.clamp_max(63)
        clean_ce = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        predicted = logits.argmax(-1)
        totals["clean_ce"] += clean_ce[mutable].sum().item()
        totals["clean_correct"] += ((predicted == tokens) & mutable).sum().item()
        totals["sites"] += mutable.sum().item()
        totals["valid_tokens"] += valid.sum().item()
        totals["invalid"] += ((AA_IDS.to(device)[predicted] != AA_IDS.to(device)[tokens]) & valid).sum().item()
        totals["exact_sequences"] += ((predicted == tokens) | ~valid).all(-1).sum().item()
        totals["sequences"] += len(tokens)
        klt = -0.5 * (1 + logvar - mu.square() - logvar.exp()).mean(-1)
        totals["kl"] += klt[valid].sum().item()
        active = mu[valid].float()
        latent_sum = active.sum(0) if latent_sum is None else latent_sum + active.sum(0)
        latent_sq = active.square().sum(0) if latent_sq is None else latent_sq + active.square().sum(0)
        latent_count += len(active)

        zero = synonymous_logits(model.decode(torch.zeros_like(mu), tokens), tokens)
        zero_ce = F.cross_entropy(zero.transpose(1, 2), targets, reduction="none")
        totals["zero_latent_ce"] += zero_ce[mutable].sum().item()
        shuffled = mu.clone()
        # Preserve amino-acid identity: otherwise shuffling can test only protein information.
        amino_acids = AA_IDS.to(device)[tokens]
        for amino_acid in range(20):
            positions = mutable & (amino_acids == amino_acid)
            values = mu[positions]
            if len(values) > 1:
                shuffled[positions] = values[torch.randperm(len(values), device=device, generator=generator)]
        shuf_logits = synonymous_logits(model.decode(shuffled, tokens), tokens)
        shuf_ce = F.cross_entropy(shuf_logits.transpose(1, 2), targets, reduction="none")
        totals["shuffled_latent_ce"] += shuf_ce[mutable].sum().item()

        corrupted, selected = corrupt(tokens, mask_probability, generator)
        masked_logits, masked_mu, masked_logvar = model(corrupted, tokens, sample=False)
        masked_logits = synonymous_logits(masked_logits, tokens)
        mce = F.cross_entropy(masked_logits.transpose(1, 2), targets, reduction="none")
        totals["masked_full_ce"] += mce[valid].sum().item()
        masked_kl = -0.5 * (1 + masked_logvar - masked_mu.square() - masked_logvar.exp()).sum(-1)
        totals["masked_kl"] += masked_kl[valid].sum().item()
        prior_logits = counts.log().expand(*tokens.shape, 64)
        prior_logits = synonymous_logits(prior_logits, tokens)
        pce = F.cross_entropy(prior_logits.transpose(1, 2), targets, reduction="none")
        totals["masked_ce"] += mce[selected].sum().item()
        totals["masked_correct"] += ((masked_logits.argmax(-1) == tokens) & selected).sum().item()
        totals["frequency_ce"] += pce[selected].sum().item()
        totals["frequency_correct"] += ((prior_logits.argmax(-1) == tokens) & selected).sum().item()
        totals["masked_sites"] += selected.sum().item()
    if totals["sites"] == 0 or totals["masked_sites"] == 0:
        raise ValueError("No evaluable degenerate codons/masked sites")
    ns, nm = totals["sites"], totals["masked_sites"]
    result = {"clean_mutable_accuracy": totals["clean_correct"] / ns,
              "clean_mutable_ce": totals["clean_ce"] / ns,
              "masked_mutable_accuracy": totals["masked_correct"] / nm,
              "masked_mutable_ce": totals["masked_ce"] / nm,
              "frequency_masked_ce": totals["frequency_ce"] / nm,
              "frequency_masked_accuracy": totals["frequency_correct"] / nm,
              "zero_latent_ce": totals["zero_latent_ce"] / ns,
              "shuffled_latent_ce": totals["shuffled_latent_ce"] / ns,
              "protein_error_rate": totals["invalid"] / totals["valid_tokens"],
              "exact_sequence_rate": totals["exact_sequences"] / totals["sequences"],
              "kl_per_dimension": totals["kl"] / totals["valid_tokens"],
              "kl_per_token": totals["kl"] / totals["valid_tokens"] * model.latent_dim,
              "masked_input_full_ce": totals["masked_full_ce"] / totals["valid_tokens"],
              "masked_input_kl_per_token": totals["masked_kl"] / totals["valid_tokens"],
              "sequences": int(totals["sequences"]), "masked_sites": int(nm)}
    variance = latent_sq / latent_count - (latent_sum / latent_count).square()
    result["active_latent_dimensions"] = int((variance > 0.01).sum().item())
    result["masked_ce_relative_gain"] = 1 - result["masked_mutable_ce"] / result["frequency_masked_ce"]
    # These are declared pilot gates, not guarantees of useful downstream optimization.
    result["decoder_pilot_gate"] = bool(result["protein_error_rate"] == 0
        and result["clean_mutable_accuracy"] >= 0.95
        and result["masked_ce_relative_gain"] >= 0.05
        and result["shuffled_latent_ce"] >= result["clean_mutable_ce"] + 0.10
        and result["zero_latent_ce"] >= result["clean_mutable_ce"] + 0.10)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="latent_decoder/config.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-development-split", action="store_true")
    parser.add_argument("--extend-epochs", type=int, help="Extend this resumed run to a larger cumulative epoch target")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume")
    mode.add_argument("--evaluate", metavar="CHECKPOINT")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()
    if args.extend_epochs is not None and not args.resume:
        raise ValueError("--extend-epochs requires --resume")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install a matching CUDA PyTorch build on the server.")
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    training = config["training"]
    lr_schedule = training.get("lr_schedule", "warmup_cosine")
    if lr_schedule not in ("constant", "warmup_cosine"):
        raise ValueError("Unknown lr_schedule")
    random.seed(training["seed"])
    torch.manual_seed(training["seed"])
    splits, audit = load_data(args.data, args.allow_development_split)
    data_directory = Path(args.data)
    data_hash = sha256(data_directory / ("manifest.json" if (data_directory / "manifest.json").exists() else "corpus.jsonl"))
    if max(r["n_codons"] for rows in splits.values() for r in rows) > config["model"]["max_codons"]:
        raise ValueError("Dataset exceeds model length; do not silently truncate")
    checkpoint = None
    if args.resume or args.evaluate:
        checkpoint = torch.load(args.resume or args.evaluate, map_location="cpu", weights_only=False)
        if checkpoint["corpus_sha256"] != data_hash or checkpoint["codons"] != CODONS:
            raise ValueError("Checkpoint data/vocabulary mismatch")
        if checkpoint["config"] != config:
            raise ValueError("Checkpoint configuration mismatch; use its config.json")
        if args.extend_epochs is not None:
            if args.extend_epochs <= training["epochs"]:
                raise ValueError("Extension target must exceed the checkpoint epoch target")
            if checkpoint["optimizer"]["param_groups"][0]["lr"] <= 0:
                raise ValueError("Checkpoint LR reached zero; a new nonzero continuation schedule must be chosen")
            training["epochs"] = args.extend_epochs
    model = CodonVMAE(**config["model"]).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    counts = codon_frequency(splits["train"])
    output = Path(args.output)
    if args.evaluate:
        if output.exists():
            raise FileExistsError("Evaluation output already exists")
        metrics = evaluate(model, splits[args.split], counts, device, training["batch_size"],
                           training["mask_probability"])
        metrics.update(split=args.split, corpus_sha256=data_hash,
                       homology_certified=audit["homology_certified"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics), flush=True)
        return
    if output.exists() and not args.resume:
        raise FileExistsError("Output exists; select a new run directory or use --resume")
    if args.resume and (Path(args.resume).resolve() != (output / "last.pt").resolve()):
        raise ValueError("Resume from this run directory's last.pt to preserve its best checkpoint and log")
    print(json.dumps(preflight(model, splits["train"], device, training["batch_size"],
                               training["mask_probability"])), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    optimizer = torch.optim.AdamW(model.parameters(), lr=training["learning_rate"],
                                  weight_decay=training["weight_decay"])
    accum = training["gradient_accumulation"]
    n_batches = math.ceil(len(splits["train"]) / training["batch_size"])
    steps_per_epoch = math.ceil(n_batches / accum)
    total_steps = steps_per_epoch * training["epochs"]
    warmup = max(1, int(total_steps * training["warmup_fraction"]))
    extension = checkpoint.get("lr_extension") if checkpoint else None
    kl_warmup_steps = (checkpoint.get("kl_warmup_steps", total_steps * training["kl_warmup_fraction"])
                       if checkpoint else total_steps * training["kl_warmup_fraction"])
    if args.extend_epochs is not None:
        current_lr = checkpoint["optimizer"]["param_groups"][0]["lr"]
        if current_lr <= 0:
            raise ValueError("Checkpoint LR reached zero; a new nonzero continuation schedule must be chosen")
        extension = dict(start_step=checkpoint["step"], end_step=total_steps,
                         start_factor=current_lr / training["learning_rate"])
        kl_warmup_steps = checkpoint.get("kl_warmup_steps",
            steps_per_epoch * checkpoint["config"]["training"]["epochs"] * training["kl_warmup_fraction"])

    def schedule(step):
        return learning_rate_factor(step, lr_schedule, warmup, total_steps, extension)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    use_amp = bool(training["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch, step, best, stale = 0, 0, float("inf"), 0
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, step, best, stale = (checkpoint[k] for k in ("next_epoch", "step", "best", "stale"))
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        # Discard logs from an interrupted epoch that will be replayed from last.pt.
        for name in ("metrics.jsonl", "steps.jsonl"):
            path = output / name
            if path.exists():
                kept = [line for line in path.read_text(encoding="utf-8").splitlines()
                        if json.loads(line)["epoch"] <= start_epoch]
                path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    if args.extend_epochs is not None:
        print(json.dumps({"event": "epochs_extended", "from_epochs": checkpoint["config"]["training"]["epochs"],
                          "to_epochs": training["epochs"], "resume_epoch": start_epoch + 1,
                          "resume_step": step, "resume_lr": optimizer.param_groups[0]["lr"],
                          "lr_extension": extension, "kl_warmup_steps": kl_warmup_steps}), flush=True)
    print(json.dumps({"parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
                      "train": len(splits["train"]), "val": len(splits["val"]),
                      "epochs": training["epochs"], "early_stopping": training["patience"] > 0,
                      "homology_certified": audit["homology_certified"], "torch": torch.__version__}), flush=True)
    for epoch in range(start_epoch, training["epochs"]):
        begin = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, count = 0., 0
        term_totals = dict(reconstruction_ce=0., masked_ce=0., kl_per_token=0.)
        window_loss, window_count = 0., 0
        window_terms = dict.fromkeys(term_totals, 0.)
        window_kl_loss = 0.
        mode_counts, mode_losses = [0, 0], [0., 0.]
        epoch_lr = optimizer.param_groups[0]["lr"]
        for index, tokens in enumerate(batches(splits["train"], training["batch_size"], shuffle=True)):
            tokens = tokens.to(device)
            probability = 0. if random.random() < training["clean_batch_probability"] else training["mask_probability"]
            corrupted, selected = corrupt(tokens, probability)
            beta = training["beta_kl"] * min(1., (step + 1) / max(1, kl_warmup_steps))
            with torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext():
                logits, mu, logvar = model(corrupted, tokens)
                loss, terms = vae_loss(logits, tokens, mu, logvar, selected, beta,
                                   training["masked_loss_weight"], training.get("full_loss_weight", 1.0))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; inspect before resuming")
            group_size = min(accum, n_batches - (index // accum) * accum)
            scaler.scale(loss / group_size).backward()
            loss_value = loss.detach().item()
            running_loss += loss_value
            window_loss += loss_value
            window_count += 1
            mode = int(probability > 0)
            mode_counts[mode] += 1
            mode_losses[mode] += loss_value
            for name in term_totals:
                value = terms[name].item()
                term_totals[name] += value
                window_terms[name] += value
                if name == "kl_per_token":
                    window_kl_loss += beta * value
            count += 1
            if (index + 1) % accum == 0 or index + 1 == n_batches:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"],
                                                     error_if_nonfinite=not use_amp)
                scaler.step(optimizer)
                scaler.update()
                if torch.isfinite(norm):
                    scheduler.step()
                    step += 1
                optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % training.get("log_every_steps", 10) == 0 or index + 1 == n_batches:
                    progress = {"epoch": epoch + 1, "step": step, "batch": count, "batches": n_batches,
                                "loss": window_loss / window_count, "epoch_mean_loss": running_loss / count,
                                "beta": beta, "lr": epoch_lr, "grad_norm": norm.item(),
                                "full_ce": window_terms["reconstruction_ce"] / window_count,
                                "masked_ce": window_terms["masked_ce"] / window_count,
                                "kl_per_token": window_terms["kl_per_token"] / window_count,
                                "weighted_kl": window_kl_loss / window_count,
                                "clean_batches": mode_counts[0], "masked_batches": mode_counts[1],
                                "masked_batch_fraction": mode_counts[1] / window_count,
                                "clean_mode_loss": mode_losses[0] / mode_counts[0] if mode_counts[0] else None,
                                "masked_mode_loss": mode_losses[1] / mode_counts[1] if mode_counts[1] else None}
                    with (output / "steps.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(progress) + "\n")
                    print(json.dumps(progress), flush=True)
                    window_loss, window_count = 0., 0
                    window_terms = dict.fromkeys(term_totals, 0.)
                    window_kl_loss = 0.
                    mode_counts, mode_losses = [0, 0], [0., 0.]
                epoch_lr = optimizer.param_groups[0]["lr"]
        metrics = evaluate(model, splits["val"], counts, device, training["batch_size"], training["mask_probability"])
        # Selection uses validation only; test is a separate explicit command.
        selection = metrics["clean_mutable_ce"] + metrics["masked_mutable_ce"]
        improved = selection < best
        best, stale = (selection, 0) if improved else (best, stale + 1)
        metrics.update(epoch=epoch + 1, step=step, train_loss=running_loss/count,
                       train_full_ce=term_totals["reconstruction_ce"]/count,
                       train_masked_ce=term_totals["masked_ce"]/count,
                       train_kl_per_token=term_totals["kl_per_token"]/count,
                       learning_rate=epoch_lr, beta=beta,
                       epoch_seconds=time.perf_counter()-begin, selection_loss=selection,
                       peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
        metrics["val_loss_fixed_beta"] = (training.get("full_loss_weight", 1.0) * metrics["masked_input_full_ce"]
            + training["masked_loss_weight"] * metrics["masked_mutable_ce"]
            + training["beta_kl"] * metrics["masked_input_kl_per_token"])
        if any(isinstance(value, (int, float)) and not math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Nonfinite validation/training metrics; inspect before resuming")
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "config": config,
                 "corpus_sha256": data_hash, "codons": CODONS, "next_epoch": epoch + 1,
                 "step": step, "best": best, "stale": stale, "python_rng": random.getstate(),
                 "lr_extension": extension, "kl_warmup_steps": kl_warmup_steps,
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}
        temporary = output / "checkpoint.tmp"
        torch.save(state, temporary)
        temporary.replace(output / "last.pt")
        if improved:
            torch.save(state, temporary)
            temporary.replace(output / "best.pt")
        print(json.dumps(metrics), flush=True)
        if training["patience"] > 0 and stale >= training["patience"]:
            print("Early stopping on validation only", flush=True)
            break


if __name__ == "__main__":
    main()
