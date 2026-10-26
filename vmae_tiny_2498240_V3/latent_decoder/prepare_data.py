"""Prepare a natural CDS corpus; never load optimization output as training data."""
import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import zipfile


GENETIC_CODE = dict(zip(map("".join, itertools.product("TCAG", repeat=3)),
    "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"))
EXPECTED_MD5 = "9e73fdb5b25aa063a8152b541ac302d5"
SEQUENCE_MEMBER = "human_coding_transcript_sequences.csv"


def digest_file(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def translate(cds):
    if not cds or set(cds) - set("ACGT"):
        return None, "non_acgt"
    if len(cds) % 3:
        return None, "incomplete_codon"
    if not cds.startswith("ATG"):
        return None, "non_atg_start"
    if cds[-3:] not in {"TAA", "TAG", "TGA"}:
        return None, "missing_terminal_stop"
    aa = "".join(GENETIC_CODE[cds[i:i + 3]] for i in range(0, len(cds), 3))
    if "*" in aa[:-1]:
        return None, "internal_stop"
    return aa[:-1], None


def rows(archive, name):
    with archive.open(name) as raw:
        yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))


def reference_cds(path):
    result = {}
    header, sequence = None, []
    def save():
        match = re.search(r"CDS:(\d+)-(\d+)", header)
        if match:
            result[header.split("|")[0]] = "".join(sequence)[int(match[1]) - 1:int(match[2])].upper()
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if header:
                    save()
                header, sequence = line[1:].strip(), []
            else:
                sequence.append(line.strip())
        if header:
            save()
    return result


class Components:
    def __init__(self, size):
        self.parent = list(range(size))

    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        a, b = self.root(a), self.root(b)
        self.parent[max(a, b)] = min(a, b)


def build(args):
    source = Path(args.official_zip)
    if source.stat().st_size != 121954830 or digest_file(source, "md5") != EXPECTED_MD5:
        raise ValueError("Official source size/MD5 mismatch; partial downloads are not valid inputs")
    excluded = Counter()
    records = {}
    protected_genes = set()
    annotation = Path("data/reference/gencode_v28/gencode.v28.pc_transcripts.cds_index.tsv")
    reference_path = annotation.with_name("gencode.v28.pc_transcripts.fa.gz")
    reference = reference_cds(reference_path)
    restored_stops = 0
    with annotation.open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["transcript_id"].split(".")[0] == "ENST00000372522":
                protected_genes.add(row["gene_id"].split(".")[0])
    official_train, official_test = set(), set()
    with zipfile.ZipFile(source) as archive:
        for row in rows(archive, "genes.csv"):
            if row.get("Training_set"):
                official_train.add(row["Training_set"].split(".")[0])
            if row.get("Test_set"):
                official_test.add(row["Test_set"].split(".")[0])
        if not official_train or not official_test or official_train & official_test:
            raise ValueError("Missing or overlapping official gene split")
        official_genes = official_train | official_test
        major_isoforms = {row["isoform_id"] for row in rows(archive, "mjIsoform_320.csv")}
        source_rows = 0
        for row in rows(archive, SEQUENCE_MEMBER):
            source_rows += 1
            gene, tx = row["gene_id"].split(".")[0], row["transcript_id"]
            if tx.split(".")[0] == "ENST00000372522":
                protected_genes.add(gene)
            if tx not in major_isoforms:
                excluded["not_author_major_isoform"] += 1
                continue
            if gene not in official_genes:
                excluded["not_in_official_gene_split"] += 1
                continue
            cds = row["cds_sequence"].upper()
            ref = reference.get(tx, "")
            # GTF CDS features omit the stop; restore only an exact reference-verified stop.
            if cds[-3:] not in {"TAA", "TAG", "TGA"} and ref[-3:] in {"TAA", "TAG", "TGA"} and cds == ref[:-3]:
                cds = cds + ref[-3:]
                restored_stops += 1
            aa, reason = translate(cds)
            if reason:
                excluded[reason] += 1
                continue
            n_codons = len(cds) // 3
            if n_codons < args.min_codons or n_codons > args.max_codons:
                excluded["length_out_of_range"] += 1
                continue
            if cds not in records:
                records[cds] = {"sequence_id": "cds_" + hashlib.sha256(cds.encode()).hexdigest(),
                    "transcript_ids": set(), "gene_ids": set(), "protein_sequence": aa,
                    "cds_dna": cds, "n_codons": n_codons,
                    "source": "figshare:10.6084/m9.figshare.28916288.v3/" + SEQUENCE_MEMBER + ";stop_annotation:gencode_v28"}
            records[cds]["transcript_ids"].add(tx)
            records[cds]["gene_ids"].add(gene)
    if not records or not protected_genes:
        raise ValueError("Empty corpus or protected SVBP transcript absent from annotation")
    data = sorted(records.values(), key=lambda row: row["sequence_id"])
    components = Components(len(data))
    owner = {}
    for i, row in enumerate(data):
        row["transcript_ids"] = sorted(row["transcript_ids"])
        row["gene_ids"] = sorted(row["gene_ids"])
        for key in ["aa:" + row["protein_sequence"]] + ["gene:" + g for g in row["gene_ids"]]:
            if key in owner:
                components.union(i, owner[key])
            else:
                owner[key] = i
    ids = {row["sequence_id"]: i for i, row in enumerate(data)}
    cluster_hash = None
    if args.clusters:
        cluster_path = Path(args.clusters)
        cluster_hash = digest_file(cluster_path)
        covered = set()
        with cluster_path.open(encoding="utf-8-sig") as f:
            for line in f:
                rep, member = line.rstrip("\r\n").split("\t")
                if rep not in ids or member not in ids:
                    raise ValueError("Cluster TSV contains IDs outside this exact corpus")
                components.union(ids[rep], ids[member])
                covered.update((rep, member))
        if covered != set(ids):
            raise ValueError("Cluster TSV must cover every corpus ID including singleton self-pairs")
    protected_proteins = set()
    protected_paths = []
    for name in ("gluc", "svbp", "NGF", "H1N1"):
        path = Path("data") / (name + "_cds") / "train_data.txt"
        if path.exists():
            cds = path.read_text(encoding="utf-8-sig").strip().upper()
            aa, reason = translate(cds)
            if reason:
                raise ValueError("Invalid protected WT sequence: " + str(path))
            protected_proteins.add(aa)
            protected_paths.append({"path": str(path), "sha256": digest_file(path)})
    groups = defaultdict(list)
    for i in range(len(data)):
        groups[components.root(i)].append(i)
    counts = Counter()
    protection_counts = Counter()
    for indices in groups.values():
        group = [data[i] for i in indices]
        component_id = "component_" + hashlib.sha256("\n".join(sorted(r["sequence_id"] for r in group)).encode()).hexdigest()
        genes = {g for r in group for g in r["gene_ids"]}
        official_holdout = bool(genes & official_test)
        target_holdout = bool(genes & protected_genes) or any(r["protein_sequence"] in protected_proteins for r in group)
        if official_holdout or target_holdout:
            split = "test"
            protection_counts["official_test_components"] += int(official_holdout)
            protection_counts["protected_wt_components"] += int(target_holdout)
        else:
            split = "val" if int(hashlib.sha256(("latent-decoder-v1:" + component_id).encode()).hexdigest(), 16) % 10 == 0 else "train"
        for row in group:
            row["cluster_id"], row["split"] = component_id, split
            counts[split] += 1
    # These assertions verify separation after all transitive gene/protein unions.
    for field in ("gene_ids", "transcript_ids", "protein_sequence"):
        seen = {}
        for row in data:
            values = [row[field]] if field == "protein_sequence" else row[field]
            for value in values:
                assert value not in seen or seen[value] == row["split"], field + " crossed splits"
                seen[value] = row["split"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    targets = [output / name for name in ("corpus.jsonl", "proteins.fasta", "audit.json")]
    if any(p.exists() for p in targets):
        raise ValueError("Output already exists; choose a new --output directory")
    with targets[0].open("w", encoding="utf-8", newline="\n") as f, targets[1].open("w", encoding="utf-8", newline="\n") as proteins:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            proteins.write(">" + row["sequence_id"] + "\n" + row["protein_sequence"] + "\n")
    audit = {"source_verified": True, "homology_certified": bool(args.clusters),
        "split_status": "external-homology-clusters-supplied" if args.clusters else "not-homology-certified",
        "source_zip": str(source), "source_md5": EXPECTED_MD5, "source_member": SEQUENCE_MEMBER,
        "source_rows": source_rows, "author_major_isoforms": len(major_isoforms),
        "accepted_transcript_rows": source_rows - sum(excluded.values()),
        "collapsed_duplicate_cds_rows": source_rows - sum(excluded.values()) - len(data),
        "reference_verified_stop_restorations": restored_stops,
        "stop_reference": {"path": str(reference_path), "sha256": digest_file(reference_path)},
        "official_gene_counts": {"train": len(official_train), "test": len(official_test)},
        "counts": dict(counts), "unique_cds": len(data), "unique_proteins": len({r['protein_sequence'] for r in data}),
        "components": len(groups), "excluded_rows": dict(excluded), "min_codons": args.min_codons,
        "max_codons": args.max_codons, "n_codons_includes_stop": True,
        "length_nt": {"min": min(len(r["cds_dna"]) for r in data),
                      "median": sorted(len(r["cds_dna"]) for r in data)[len(data) // 2],
                      "max": max(len(r["cds_dna"]) for r in data)},
        "protected_genes": sorted(protected_genes), "protected_wt_sources": protected_paths,
        "protected_gene_annotation": {"path": str(annotation), "sha256": digest_file(annotation)},
        "holdout_components": dict(protection_counts), "clusters_sha256": cluster_hash,
        "cluster_note": "Without clusters, exact-AA/gene separation only. Supplied cluster methodology/identity/coverage must be recorded and checked separately.",
        "corpus_sha256": digest_file(targets[0]), "proteins_sha256": digest_file(targets[1]),
        "no_expression_selection": True, "optimization_sequences_used_for_training": False}
    targets[2].write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


def self_check():
    assert translate("ATGGAATAA") == ("ME", None)
    assert translate("ATGTAATAA")[1] == "internal_stop"
    assert translate("ATGNAA")[1] == "non_acgt"
    assert translate("ATGGAA")[1] == "missing_terminal_stop"
    assert translate("ATGTA")[1] == "incomplete_codon"
    assert translate("TTGGAATAA")[1] == "non_atg_start"
    c = Components(4)
    c.union(2, 3)
    c.union(0, 2)
    assert c.root(0) == c.root(3) and c.root(1) != c.root(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-zip", default="data/official/dataset_for_translation_model.zip")
    parser.add_argument("--output", default="data/latent_decoder")
    parser.add_argument("--clusters", help="MMseqs createtsv representative/member pairs for this proteins.fasta")
    parser.add_argument("--min-codons", type=int, default=30)
    parser.add_argument("--max-codons", type=int, default=1500)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if not args.self_check:
        if not 1 <= args.min_codons <= args.max_codons:
            parser.error("Require 1 <= min-codons <= max-codons")
        build(args)
