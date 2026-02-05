from __future__ import annotations

from typing import List, Optional, Tuple


def build_attention_and_labels(
    sequences,
    *,
    prompt_lens: List[int],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
):
    """
    Build (attention_mask, labels, completion_lens) for CausalLM training.

    - Mask out prompt tokens and any padding beyond the first EOS.
    - completion_lens counts tokens after the prompt (at least 1 to avoid div-by-zero).
    """
    import torch

    if sequences.ndim != 2:
        raise ValueError("sequences must be [B, L]")
    bsz, seqlen = sequences.shape
    if len(prompt_lens) != bsz:
        raise ValueError("prompt_lens length mismatch")

    attention_mask = torch.zeros((bsz, seqlen), dtype=torch.long, device=sequences.device)
    labels = sequences.clone()
    completion_lens = torch.zeros((bsz,), dtype=torch.long, device=sequences.device)

    for i in range(bsz):
        p = int(prompt_lens[i])
        p = max(0, min(p, seqlen))

        end = seqlen
        seq = sequences[i]

        # Prefer first EOS after prompt as the natural end.
        if eos_token_id is not None:
            hits = (seq[p:] == int(eos_token_id)).nonzero(as_tuple=False)
            if hits.numel() > 0:
                end = min(end, p + int(hits[0].item()) + 1)  # include EOS token

        # If a real PAD token exists (and differs from EOS), trim at first PAD too.
        if pad_token_id is not None and pad_token_id != eos_token_id:
            hits = (seq[p:] == int(pad_token_id)).nonzero(as_tuple=False)
            if hits.numel() > 0:
                end = min(end, p + int(hits[0].item()))

        attention_mask[i, :end] = 1
        labels[i, :p] = -100
        labels[i, end:] = -100

        completion_lens[i] = max(1, end - p)

    return attention_mask, labels, completion_lens


def per_sample_nll(logits, labels) -> "torch.Tensor":
    """
    Compute per-sample NLL summed over non-ignored tokens (labels != -100).
    """
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.size(-1)

    flat = F.cross_entropy(
        shift_logits.view(-1, vocab),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    return flat.view(labels.size(0), -1).sum(dim=1)


def weighted_rl_loss(
    model,
    sequences,
    *,
    prompt_lens: List[int],
    weights: List[float],
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> Tuple["torch.Tensor", dict]:
    """
    Policy-gradient style loss for language models:

        loss = mean_i [ w_i * NLL_i / completion_len_i ]

    where w_i is a (possibly signed) advantage weight.
    """
    import torch

    attn, labels, comp_lens = build_attention_and_labels(
        sequences,
        prompt_lens=prompt_lens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    out = model(input_ids=sequences, attention_mask=attn)
    nll_sum = per_sample_nll(out.logits, labels)

    w = torch.tensor(weights, device=sequences.device, dtype=nll_sum.dtype)
    loss_per = w * (nll_sum / comp_lens.to(nll_sum.dtype))
    loss = loss_per.mean()

    metrics = {
        "mean_nll": float((nll_sum / comp_lens.to(nll_sum.dtype)).mean().item()),
        "mean_weight": float(w.mean().item()),
        "mean_abs_weight": float(w.abs().mean().item()),
    }
    return loss, metrics

