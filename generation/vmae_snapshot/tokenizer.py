"""Shared, dependency-free CDS tokenizer; IDs 0..63 match original RiboDecode."""

CODONS = ("TTT TTC TTA TTG CTT CTC CTA CTG ATT ATC ATA ATG GTT GTC GTA GTG "
          "TAT TAC TAA TAG CAT CAC CAA CAG AAT AAC AAA AAG GAT GAC GAA GAG "
          "TCT TCC TCA TCG CCT CCC CCA CCG ACT ACC ACA ACG GCT GCC GCA GCG "
          "TGT TGC TGA TGG CGT CGC CGA CGG AGT AGC AGA AGG GGT GGC GGA GGG").split()
AA = list("FFLLLLLLIIIMVVVVYY**HHQQNNKKDDEESSSSPPPPTTTTAAAACC*WRRRRSSRRGGGG")
PAD, MASK = 64, 65
TOKEN = {codon: i for i, codon in enumerate(CODONS)}


def encode_cds(sequence):
    sequence = sequence.upper().replace("U", "T")
    if len(sequence) % 3 or not sequence or set(sequence) - set("ACGT"):
        raise ValueError("CDS must contain complete unambiguous codons")
    ids = [TOKEN[sequence[i:i+3]] for i in range(0, len(sequence), 3)]
    protein = "".join(AA[i] for i in ids)
    if not sequence.startswith("ATG") or protein[-1] != "*" or "*" in protein[:-1]:
        raise ValueError("CDS must start with ATG and contain only one terminal stop")
    return ids


def decode_cds(ids, rna=False):
    if not ids or any(type(i) is not int or not 0 <= i < 64 for i in ids):
        raise ValueError("A complete CDS contains only codon IDs 0..63; remove padding first")
    sequence = "".join(CODONS[i] for i in ids)
    encode_cds(sequence)
    return sequence.replace("T", "U") if rna else sequence


def specification():
    return {"version": "ribodecode-codon-v1", "codon_to_id": TOKEN,
            "id_to_codon": CODONS, "id_to_amino_acid": AA,
            "special_tokens": {"PAD": PAD, "MASK": MASK},
            "vocab_size": 66, "genetic_code": "standard nuclear",
            "input_normalization": "uppercase; U -> T", "includes_terminal_stop": True,
            "padding_stored_in_records": False, "truncation": False,
            "legacy_NNN": {"id": 64, "note": "RiboDecode scorer only; not a CDS token or VMAE PAD. Append a zero NNN probability channel at valid codon positions when adapting to its 65-channel scorer."}}
