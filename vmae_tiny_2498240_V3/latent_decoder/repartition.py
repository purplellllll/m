"""Repartition a verified prepared corpus using supplied protein clusters, without raw data."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

try:
    from .prepare_data import Components, digest_file, translate
except ImportError:
    from prepare_data import Components, digest_file, translate


SPLITS = ("train", "val", "test")


def read_fasta(path):
    proteins, name, pieces = {}, None, []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    proteins[name] = "".join(pieces)
                name, pieces = line[1:], []
                if not name or any(c.isspace() for c in name) or name in proteins:
                    raise ValueError("Invalid or duplicate FASTA identifier")
            elif name is None:
                raise ValueError("FASTA sequence precedes its identifier")
            else:
                pieces.append(line)
    if name is not None:
        proteins[name] = "".join(pieces)
    return proteins


def load_prepared(directory):
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    if audit.get("source_verified") is not True:
        raise ValueError("Original audit must declare source_verified=true")
    for filename, key in (("corpus.jsonl", "corpus_sha256"), ("proteins.fasta", "proteins_sha256")):
        if audit.get(key) != digest_file(directory / filename):
            raise ValueError("Original " + filename + " checksum mismatch")
    records, ids, cds_seen = [], set(), set()
    with (directory / "corpus.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            identifier = record.get("sequence_id")
            if not isinstance(identifier, str) or not identifier or any(c.isspace() for c in identifier) or ">" in identifier or identifier in ids:
                raise ValueError("Invalid or duplicate sequence_id")
            cds = record.get("cds_dna")
            if not isinstance(cds, str):
                raise ValueError("Missing CDS")
            protein, reason = translate(cds)
            if reason or protein != record.get("protein_sequence"):
                raise ValueError("Invalid CDS/protein annotation: " + str(reason))
            count = len(cds) // 3
            if type(record.get("n_codons")) is not int or record["n_codons"] != count:
                raise ValueError("Invalid codon count")
            if not audit.get("min_codons", 1) <= count <= audit.get("max_codons", 1500):
                raise ValueError("CDS outside audited length limits")
            if cds in cds_seen or record.get("split") not in SPLITS:
                raise ValueError("Duplicate CDS or invalid original split")
            for field in ("gene_ids", "transcript_ids"):
                values = record.get(field)
                if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values) or len(set(values)) != len(values):
                    raise ValueError("Invalid " + field)
            if not isinstance(record.get("cluster_id"), str) or not record["cluster_id"]:
                raise ValueError("Missing original component ID")
            if not isinstance(record.get("source"), str) or not record["source"]:
                raise ValueError("Missing source provenance")
            ids.add(identifier)
            cds_seen.add(cds)
            records.append(record)
    if not records:
        raise ValueError("Empty prepared corpus")
    if read_fasta(directory / "proteins.fasta") != {r["sequence_id"]: r["protein_sequence"] for r in records}:
        raise ValueError("Protein FASTA does not match corpus records")
    actual = Counter(r["split"] for r in records)
    if any(audit.get("counts", {}).get(s, 0) != actual[s] for s in SPLITS):
        raise ValueError("Original split counts do not match audit")
    if audit.get("unique_cds", len(records)) != len(records):
        raise ValueError("Original record count does not match audit")
    return sorted(records, key=lambda r: r["sequence_id"]), audit


def read_clusters(path, ids):
    membership = {}
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 2:
                raise ValueError("Cluster TSV requires exactly representative/member columns")
            representative, member = fields
            if representative not in ids or member not in ids:
                raise ValueError("Cluster contains unknown corpus ID")
            if member in membership:
                raise ValueError("Duplicate cluster member: " + member)
            membership[member] = representative
    if set(membership) != ids:
        raise ValueError("Cluster TSV must cover every sequence, including singleton self-pairs")
    if any(membership[representative] != representative for representative in membership.values()):
        raise ValueError("Representative must have its own self-pair in the same cluster")
    return membership


def repartition(data, clusters, output):
    data, clusters, output = Path(data), Path(clusters), Path(output)
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    records, original_audit = load_prepared(data)
    ids = {r["sequence_id"]: i for i, r in enumerate(records)}
    membership = read_clusters(clusters, set(ids))
    components, owners = Components(len(records)), {}
    old_test = {r["sequence_id"] for r in records if r["split"] == "test"}
    for i, record in enumerate(records):
        keys = [("protein", record["protein_sequence"]), ("cluster", membership[record["sequence_id"]]),
                ("original_component", record["cluster_id"])]
        keys += [("gene", g) for g in record["gene_ids"]]
        keys += [("transcript", t) for t in record["transcript_ids"]]
        for key in keys:
            if key in owners:
                components.union(i, owners[key])
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i in range(len(records)):
        groups[components.root(i)].append(i)
    for indices in groups.values():
        members = [records[i]["sequence_id"] for i in indices]
        component = "component_" + hashlib.sha256("\n".join(sorted(members)).encode()).hexdigest()
        if old_test.intersection(members):
            split = "test"
        else:
            split = "val" if int(hashlib.sha256(("latent-decoder-v1:" + component).encode()).hexdigest(), 16) % 10 == 0 else "train"
        for i in indices:
            records[i]["split"], records[i]["cluster_id"] = split, component
    assert all(r["split"] == "test" for r in records if r["sequence_id"] in old_test)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "corpus.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    for split in (None,) + SPLITS:
        name = "proteins.fasta" if split is None else split + "_proteins.fasta"
        with (output / name).open("x", encoding="utf-8", newline="\n") as handle:
            for record in records:
                if split is None or record["split"] == split:
                    handle.write(">" + record["sequence_id"] + "\n" + record["protein_sequence"] + "\n")
    counts = Counter(r["split"] for r in records)
    audit = dict(original_audit)
    audit.update({"source_verified": True, "homology_certified": True,
        "split_status": "external-homology-clusters-supplied", "homology_search_audited": False,
        "cluster_note": "Complete external cluster membership supplied; methodology/thresholds and cross-split sensitive search require separate audit. Not a zero-homology guarantee.",
        "original_data_directory": str(data), "original_audit_sha256": digest_file(data / "audit.json"),
        "original_corpus_sha256": original_audit["corpus_sha256"],
        "original_proteins_sha256": original_audit["proteins_sha256"],
        "original_audit": original_audit, "clusters_sha256": digest_file(clusters),
        "cluster_tsv": str(clusters), "counts": {s: counts[s] for s in SPLITS},
        "components": len(groups), "original_test_records": len(old_test),
        "test_records_added_by_components": counts["test"] - len(old_test),
        "split_policy": "Union gene, transcript, exact protein, supplied cluster and old component; any old test member forces whole component to test; remaining component hash mod 10 == 0 goes to val.",
        "corpus_sha256": digest_file(output / "corpus.jsonl"),
        "proteins_sha256": digest_file(output / "proteins.fasta"),
        "split_proteins_sha256": {s: digest_file(output / (s + "_proteins.fasta")) for s in SPLITS}})
    (output / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/latent_decoder")
    parser.add_argument("--clusters", required=True, help="Complete MMseqs representative/member TSV")
    parser.add_argument("--output", required=True, help="New output directory; never overwritten")
    args = parser.parse_args()
    result = repartition(args.data, args.clusters, args.output)
    print(json.dumps({k: result[k] for k in ("counts", "components", "homology_certified", "homology_search_audited")}, indent=2))
