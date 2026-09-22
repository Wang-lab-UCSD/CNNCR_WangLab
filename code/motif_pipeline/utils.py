"""
Shared utility functions for the motif processing pipeline.

Covers:
  - Cell line list loading
  - MEME file I/O (parsing and writing)
  - PWM statistics: information content, Pearson distance, reverse complement
  - Motif orientation normalisation and representative-motif selection
  - TOMTOM result loading and network-based motif clustering
"""

import numpy as np
import pandas as pd
import networkx as nx


# ── Cell line helpers ─────────────────────────────────────────────────────────

def load_bssids(bssid_file, bad_qc=("BSS01297", "BSS01325")):
    """
    Load the non-redundant BSSID list and drop known bad-QC samples.
    Returns a numpy array of BSSID strings.
    """
    bssids = np.genfromtxt(bssid_file, delimiter="\n", dtype=str)
    return np.array([b for b in bssids if b not in bad_qc])


# ── MEME file I/O ─────────────────────────────────────────────────────────────

def parse_meme_file(meme_file):
    """
    Parse a MEME-format file and return all PWMs as NumPy arrays.

    Returns
    -------
    motifs      : list of (L, 4) float arrays, one per motif
    motif_names : list of motif name strings (second token on each MOTIF line)
    """
    motifs, motif_names = [], []
    current_pwm, current_name = [], None
    reading = False

    with open(meme_file) as fh:
        for line in fh:
            if line.startswith("MOTIF"):
                if current_pwm:                          # save previous motif
                    motifs.append(np.array(current_pwm))
                    motif_names.append(current_name)
                    current_pwm = []
                parts = line.strip().split()
                current_name = parts[1] if len(parts) >= 2 else f"motif_{len(motifs)}"
                reading = False
            elif line.startswith("letter-probability matrix"):
                reading = True
                current_pwm = []
            elif reading and line.strip():
                current_pwm.append([float(x) for x in line.strip().split()])

    if current_pwm:                                      # save last motif
        motifs.append(np.array(current_pwm))
        motif_names.append(current_name)

    return motifs, motif_names


def write_meme_file(motifs, file_path, motif_names=None):
    """
    Write a list of (L, 4) PWMs to a MEME-format file.

    MOTIF header format is "<name> <name>" so that TOMTOM treats the first
    token as both the display name and the matching key.
    motif_names defaults to sequential integers when not provided.
    """
    with open(file_path, "w") as fh:
        fh.write("MEME version 4.1.1\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        for i, motif in enumerate(motifs):
            name = str(motif_names[i]) if motif_names is not None else str(i)
            fh.write(f"MOTIF {name} {name}\n")
            fh.write(
                f"letter-probability matrix: alength=4 w={motif.shape[0]} "
                f"nsites=1 E=0.0e+00\n"
            )
            for row in motif:
                fh.write("\t".join(str(x) for x in row) + "\n")
            fh.write("\n")


# ── PWM statistics ────────────────────────────────────────────────────────────

def information_content(pwm, background=0.25):
    """
    Total information content of a PWM (summed over all positions and bases).
    Uses uniform background frequency by default (0.25 for DNA).
    Higher values indicate more conserved / informative motifs.
    """
    ic = 0.0
    for position in pwm:
        for p in position:
            if p > 0:
                ic += p * np.log2(p / background)
    return ic


def reverse_complement_pwm(pwm):
    """
    Reverse complement of a (L, 4) PWM.
    Column order: A=0, C=1, G=2, T=3 → complement map is [3, 2, 1, 0].
    """
    return pwm[::-1][:, [3, 2, 1, 0]]


def pearson_distance(pwm1, pwm2):
    """
    Mean per-position Pearson distance (1 − correlation) between two PWMs.
    Both PWMs must have the same length L. Returns a float in [0, 2].
    Lower = more similar.
    """
    assert pwm1.shape[0] == pwm2.shape[0], (
        f"PWM length mismatch: {pwm1.shape[0]} vs {pwm2.shape[0]}"
    )
    return float(np.mean(
        [1 - np.corrcoef(pwm1[i], pwm2[i])[0, 1] for i in range(pwm1.shape[0])]
    ))


# ── Motif orientation and representative selection ────────────────────────────

def normalize_orientations(motifs):
    """
    Orient each motif so that it is on the same strand as the highest-IC motif
    in the group. Returns the list unchanged for singleton groups.

    All motifs must share the same PWM length (same CNN filter size).
    """
    if len(motifs) == 1:
        return motifs
    ref_idx = int(np.argmax([information_content(m) for m in motifs]))
    ref = motifs[ref_idx]
    result = []
    for m in motifs:
        rc = reverse_complement_pwm(m)
        result.append(m if pearson_distance(ref, m) <= pearson_distance(ref, rc) else rc)
    return result


def representative_motif(motifs):
    """
    Return the motif in the group that is closest (Pearson distance) to the
    component-wise median of all group members.

    Falls back to the highest-IC motif if the group contains PWMs of mixed lengths
    (can happen in cross-cell-line groups where different CNN models are involved).
    """
    if len(motifs) == 1:
        return motifs[0]

    shapes = {m.shape for m in motifs}
    if len(shapes) > 1:
        # Mixed lengths: cannot compute a common median; use highest-IC motif
        return motifs[int(np.argmax([information_content(m) for m in motifs]))]

    stack  = np.stack(motifs)           # (N, L, 4)
    median = np.median(stack, axis=0)   # (L, 4) — component-wise median
    return motifs[int(np.argmin([pearson_distance(median, m) for m in motifs]))]


# ── TOMTOM result loading and network-based clustering ─────────────────────────

def read_tomtom(tsv_file, e_value_threshold, p_value_threshold):
    """
    Load a TOMTOM text-format output file (produced with -text flag) and return
    only rows that pass both statistical thresholds.

    Query_ID and Target_ID columns are returned as-is (strings); cast to int in
    the caller when motif names are integer IDs.
    """
    df = pd.read_csv(tsv_file, sep="\t", comment="#")
    df["Optimal_offset"] = df["Optimal_offset"].astype(int)
    return df[(df["E-value"] <= e_value_threshold) & (df["p-value"] <= p_value_threshold)]


def _build_similarity_graph(tomtom_df):
    """
    Build an undirected weighted graph from TOMTOM results.
    Nodes = motif IDs, edges = significant matches, edge weight = E-value.
    """
    G = nx.Graph()
    for _, row in tomtom_df.iterrows():
        G.add_edge(row["Query_ID"], row["Target_ID"], weight=row["E-value"])
    return G


def group_by_network(tomtom_df, distance_threshold):
    """
    Cluster motifs using Dijkstra path-length in the TOMTOM similarity graph.

    Two motifs are placed in the same group only if ALL pairwise Dijkstra
    distances within the group are <= distance_threshold.  This enforces
    that every member of a cluster is "close" to every other member, not
    just transitively reachable.

    Motifs not appearing in the similarity graph (no significant TOMTOM
    matches) are kept as singleton groups.

    Returns a list of groups (each a sorted list of motif IDs), largest-first.
    """
    if tomtom_df.empty:
        return []

    G        = _build_similarity_graph(tomtom_df)
    shortest = dict(nx.all_pairs_dijkstra_path_length(G))
    visited  = set()
    groups   = []

    for node in G.nodes:
        if node in visited:
            continue
        cluster = {node}
        visited.add(node)
        queue   = [node]
        while queue:
            cur = queue.pop()
            for neighbor, dist in shortest[cur].items():
                if neighbor not in visited and all(
                    shortest[m].get(neighbor, float("inf")) <= distance_threshold
                    for m in cluster
                ):
                    cluster.add(neighbor)
                    visited.add(neighbor)
                    queue.append(neighbor)
        groups.append(cluster)

    # Add isolated nodes (no TOMTOM matches) as singletons
    all_nodes = set(tomtom_df["Query_ID"]).union(set(tomtom_df["Target_ID"]))
    for node in all_nodes:
        if node not in visited:
            groups.append({node})

    return [sorted(list(g)) for g in sorted(groups, key=len, reverse=True)]
