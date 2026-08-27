"""Per-protein PyMOL .pse sessions (supplementary visualisation resource;
not a numbered manuscript figure) — one .pse per benchmark protein.

For each protein, loads its multi-chain CIF and colours the cartoon by
within-protein ESM-1b attention rank percentile using the **plasma_r**
colormap so that the **darkest residues are the highest-attention
ones** (dark purple = top rank, bright yellow = bottom rank).
Active-site residues are shown as green spheres; allosteric-site
residues as plasma-coloured spheres.

Headless PyMOL is invoked with ``pymol -cq``. The PSE files embed all
loaded structure data, so the downstream consumer does not need a
local copy of the CIF.

Sessions are written to ``manuscript/data/pymol_sessions/`` when this script is invoked.

PyMOL is located via the ``PYMOL_BIN`` environment variable, otherwise
the ``pymol`` command on PATH. On macOS the PyMOL.app binary is
typically at ``/Applications/PyMOL.app/Contents/bin/pymol``.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from reproduction.io import load_benchmark, load_manifest, load_residue_scores
from reproduction.paths import PYMOL_SESSIONS, STRUCTURES_MULTICHAIN

OUT = PYMOL_SESSIONS


def _plasma_r_rgb(val: float) -> tuple[float, float, float]:
    """Reversed plasma — val = 1 → dark purple, val = 0 → bright yellow."""
    import matplotlib.cm as cm
    r, g, b, _ = cm.plasma_r(val)
    return r, g, b


def _pymol_bin() -> str:
    env = os.environ.get("PYMOL_BIN")
    if env: return env
    found = shutil.which("pymol")
    if found: return found
    mac_default = "/Applications/PyMOL.app/Contents/bin/pymol"
    if os.path.exists(mac_default): return mac_default
    return "pymol"   # will fail at subprocess.run, caught below


def _build_script(cif_path: Path, residue_scores: dict[int, float],
                  active: set[int], allo: set[int], output: Path) -> str:
    items = sorted(residue_scores.items())
    if not items:
        return ""
    resnums = np.array([r for r, _ in items])
    scores  = np.array([s for _, s in items], dtype=float)
    ranks = rankdata(scores, method="average")
    pct = ((ranks - 1) / (len(ranks) - 1)) if len(ranks) > 1 \
          else np.zeros_like(ranks)
    pct_map = dict(zip(resnums.tolist(), pct.tolist()))

    L: list[str] = [
        f"load {os.path.abspath(cif_path)}",
        "bg_color white",
        "hide everything",
        "show cartoon",
        "color gray80",
        "# ESM-1b attention rank percentile (plasma_r; dark purple = highest)",
    ]
    for resi, p in pct_map.items():
        r, g, b = _plasma_r_rgb(p)
        c = f"attn_{resi}"
        L.append(f"set_color {c}, [{r:.4f}, {g:.4f}, {b:.4f}]")
        L.append(f"color {c}, resi {resi}")
    if active:
        sel = "+".join(str(r) for r in sorted(active))
        L += [f"select active_sites, resi {sel}",
              "show spheres, active_sites",
              "color green, active_sites",
              "set sphere_scale, 0.75, active_sites"]
    if allo:
        sel = "+".join(str(r) for r in sorted(allo))
        L += [f"select allo_sites, resi {sel}",
              "show spheres, allo_sites",
              "set sphere_scale, 0.75, allo_sites"]

    # Slight transparency on residues that are neither active nor allosteric,
    # so the labelled positions stand out against the attention-coloured bulk.
    parts = [s for s in ("active_sites", "allo_sites")
             if (s == "active_sites" and active) or (s == "allo_sites" and allo)]
    if parts:
        L += [f"select background, not ({' or '.join(parts)})",
              "set cartoon_transparency, 0.4, background"]
    else:
        L += ["set cartoon_transparency, 0.4"]

    L += ["deselect", "orient", "zoom",
          f"save {os.path.abspath(output)}",
          "quit"]
    return "\n".join(L)


def _run_pymol(script: str, label: str) -> bool:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pml", delete=False) as f:
        f.write(script); path = f.name
    try:
        res = subprocess.run([_pymol_bin(), "-cq", path],
                              capture_output=True, text=True, timeout=180)
        if res.returncode != 0:
            print(f"  PyMOL error {label}: {res.stderr[:200]}")
            return False
        return True
    except FileNotFoundError:
        print("pymol not found — install pymol-open-source or set PYMOL_BIN")
        return False
    except subprocess.TimeoutExpired:
        print(f"  PyMOL timeout {label}"); return False
    finally:
        os.unlink(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default=None,
                    help="render only this UniProt accession")
    ap.add_argument("--limit", type=int, default=None,
                    help="render only the first N proteins")
    ap.add_argument("--cif-dir", type=Path, default=None,
                    help="override the directory of multi-chain CIFs "
                         "(default: manuscript/data/structures_multichain/)")
    args = ap.parse_args()

    cif_dir = args.cif_dir or STRUCTURES_MULTICHAIN
    OUT.mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark()
    manifest  = load_manifest()
    uni_to_pdb = dict(zip(manifest.uniprot, manifest.pdb))
    try:
        scores = load_residue_scores("ESM1b")
    except FileNotFoundError:
        print("missing cached ESM-1b residue scores")
        sys.exit(1)

    todo = sorted(benchmark)
    if args.only: todo = [args.only]
    if args.limit: todo = todo[: args.limit]

    n_ok = n_skip = 0
    for u in todo:
        if u not in scores or u not in uni_to_pdb:
            n_skip += 1; continue
        pdb = uni_to_pdb[u]
        cif = cif_dir / f"{pdb.upper()}.cif"
        if not cif.exists():
            cif = cif_dir / f"{pdb.upper()}.cif.gz"
            if not cif.exists():
                print(f"  (missing CIF for {u}_{pdb})"); n_skip += 1; continue
        output = OUT / f"{u}_{pdb}.pse"
        if output.exists():
            n_ok += 1; continue
        script = _build_script(
            cif, scores[u],
            benchmark[u]["active"], benchmark[u]["allo"], output)
        if not script:
            n_skip += 1; continue
        if _run_pymol(script, u):
            n_ok += 1
            print(f"  {u}_{pdb}.pse")
        else:
            n_skip += 1
    print(f"rendered {n_ok}/{len(todo)} sessions to {OUT}  "
          f"(skipped {n_skip})")


if __name__ == "__main__":
    main()
