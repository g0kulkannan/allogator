"""ESM attention extraction and AlloGator residue scoring."""
from __future__ import annotations

import gc
from collections.abc import Iterable

import numpy as np
import torch


def get_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve an inference device without silently changing explicit choices."""
    if isinstance(requested, torch.device):
        requested = requested.type
    name = str(requested).lower()
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA device is available")
        return torch.device("cuda")
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but no MPS device is available")
        return torch.device("mps")
    raise ValueError(f"unknown device {requested!r}; use auto, cpu, cuda, or mps")


def _load_esm1b():
    import esm
    return esm.pretrained.esm1b_t33_650M_UR50S()


def _load_esm2(size: str):
    import esm
    table = {
        "8M":   esm.pretrained.esm2_t6_8M_UR50D,
        "35M":  esm.pretrained.esm2_t12_35M_UR50D,
        "150M": esm.pretrained.esm2_t30_150M_UR50D,
        "650M": esm.pretrained.esm2_t33_650M_UR50D,
        "3B":   esm.pretrained.esm2_t36_3B_UR50D,
        "15B":  esm.pretrained.esm2_t48_15B_UR50D,
    }
    return table[size]()


def load_model(
    name: str = "ESM1b",
    esm2_size: str = "650M",
    device: str | torch.device = "auto",
):
    """Return ``(model, alphabet, n_layers, device)`` ready for inference."""
    selected_device = get_device(device)
    if name == "ESM1b":
        model, alphabet = _load_esm1b()
    elif name == "ESM2":
        model, alphabet = _load_esm2(esm2_size)
    else:
        raise ValueError(name)
    model = model.to(selected_device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    n_layers = sum(1 for _ in model.layers) if hasattr(model, "layers") else 33
    return model, alphabet, n_layers, selected_device


def _reduce_active_rows(selected_attention: np.ndarray) -> np.ndarray:
    """Average layers/heads and sum active-site query rows.

    ``selected_attention`` has shape ``(layers, heads, active, residues)``.
    Keeping this reduction in one function ensures that prediction and
    benchmark scoring use the same directional attention calculation.
    """
    selected = np.asarray(selected_attention, dtype=np.float32)
    if selected.ndim != 4:
        raise ValueError(
            "selected attention must have shape (layers, heads, active, residues)"
        )
    if selected.shape[2] == 0 or selected.shape[3] == 0:
        raise ValueError("selected attention has an empty active-site or residue axis")
    if not np.isfinite(selected).all():
        raise ValueError("attention contains a non-finite value")
    return selected.mean(axis=(0, 1)).sum(axis=0)


def attention_scores_from_tensor(
    attention: np.ndarray,
    active_residues: Iterable[int],
) -> np.ndarray:
    """Apply the manuscript scoring rule to a full ESM attention tensor.

    Residue coordinates are one-based. The input tensor must have shape
    ``(layers, heads, L+2, L+2)`` including BOS and EOS tokens.
    """
    array = np.asarray(attention)
    if array.ndim != 4 or array.shape[-2] != array.shape[-1]:
        raise ValueError(
            "attention must have shape (layers, heads, L+2, L+2)"
        )
    length = array.shape[-1] - 2
    active = sorted({int(position) for position in active_residues})
    if not active:
        raise ValueError("at least one active-site residue is required")
    if active[0] < 1 or active[-1] > length:
        raise ValueError(f"active-site residues must be within 1..{length}")
    selected = np.take(array, active, axis=-2)[..., 1:-1]
    return _reduce_active_rows(selected)


def active_site_attention_scores(
    model,
    alphabet,
    device: torch.device,
    sequence: str,
    active_residues: Iterable[int],
    n_layers: int = 33,
) -> np.ndarray:
    """Score one sequence while transferring only active query rows to CPU."""
    length = len(sequence)
    active = sorted({int(position) for position in active_residues})
    if not active:
        raise ValueError("at least one active-site residue is required")
    if active[0] < 1 or active[-1] > length:
        raise ValueError(f"active-site residues must be within 1..{length}")

    batch_converter = alphabet.get_batch_converter()
    with torch.no_grad():
        _, _, tokens = batch_converter([("seq", sequence)])
        tokens = tokens.to(device)
        result = model(
            tokens,
            repr_layers=[n_layers],
            need_head_weights=True,
            return_contacts=False,
        )
        attention = result.get("attentions")
        if attention is None or attention.ndim != 5:
            raise RuntimeError("ESM-1b did not return the expected attention tensor")
        expected = length + 2
        if attention.shape[-2:] != (expected, expected):
            raise RuntimeError(
                f"unexpected attention shape {tuple(attention.shape)} for "
                f"a {length}-residue sequence"
            )
        active_tokens = torch.as_tensor(
            active, dtype=torch.long, device=attention.device
        )
        selected = attention[0].index_select(-2, active_tokens)[..., 1:-1]
        selected_array = selected.detach().float().cpu().numpy()
        scores = _reduce_active_rows(selected_array)
        del result, attention, selected, selected_array, tokens
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    return scores


def attention_tensor(model, alphabet, device, sequence: str,
                      n_layers: int = 33) -> np.ndarray:
    """Return the full attention tensor of shape
    (n_layers, n_heads, L+2, L+2) for one sequence. The +2 axes are the
    BOS / EOS positions; callers should slice [1:-1, 1:-1] for the
    residue-only sub-block.
    """
    bc = alphabet.get_batch_converter()
    with torch.no_grad():
        _, _, tok = bc([("seq", sequence)])
        tok = tok.to(device)
        out = model(tok, repr_layers=[n_layers], return_contacts=True)
        attn = out["attentions"][0].cpu().numpy().astype(np.float32)
        del out, tok
    if device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()
    return attn


def free_model(model):
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
