"""Repetition, frequency and presence penalties for dLLM denoising.

dLLM algorithms pick tokens with argmax over the model's logits and never go
through the sampler, so the request's penalties were accepted but ignored.
This applies them to a block's logits before an algorithm reads them, with the
same arithmetic as SGLang's penalizers: repetition divides positive logits and
multiplies negative ones; frequency subtracts per occurrence; presence once per
token. Like those penalizers, only generated tokens count, not the prompt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

# (token ids, occurrence counts, repetition, frequency, presence)
_Row = Tuple[torch.Tensor, torch.Tensor, float, float, float]


def _values(sampling_params) -> Tuple[float, float, float]:
    return (
        float(sampling_params.repetition_penalty),
        float(sampling_params.frequency_penalty),
        float(sampling_params.presence_penalty),
    )


class DllmPenalties:
    def __init__(self, rows: List[Optional[_Row]]):
        self.rows = rows

    @classmethod
    def from_token_lists(
        cls,
        outputs: List[List[int]],
        params: List[Tuple[float, float, float]],
        device: torch.device,
    ) -> Optional["DllmPenalties"]:
        rows: List[Optional[_Row]] = []
        for output_ids, (rep, freq, pres) in zip(outputs, params):
            if not output_ids or (rep == 1.0 and freq == 0.0 and pres == 0.0):
                rows.append(None)
                continue
            ids, counts = torch.unique(
                torch.tensor(output_ids, dtype=torch.long), return_counts=True
            )
            rows.append(
                (
                    ids.to(device, non_blocking=True),
                    counts.to(torch.float32).to(device, non_blocking=True),
                    rep,
                    freq,
                    pres,
                )
            )
        return cls(rows) if any(row is not None for row in rows) else None

    @classmethod
    def from_reqs(cls, reqs: List["Req"], device) -> Optional["DllmPenalties"]:
        return cls.from_token_lists(
            [list(req.output_ids) for req in reqs],
            [_values(req.sampling_params) for req in reqs],
            device,
        )

    def apply(self, index: int, logits: torch.Tensor) -> None:
        """Penalize, in place, the [positions, vocab] logits of request `index`."""
        row = self.rows[index] if index < len(self.rows) else None
        if row is None:
            return
        ids, counts, rep, freq, pres = row
        selected = logits[:, ids].float()
        if rep != 1.0:
            selected = torch.where(selected > 0, selected / rep, selected * rep)
        if freq != 0.0:
            selected = selected - freq * counts
        if pres != 0.0:
            selected = selected - pres
        logits[:, ids] = selected.to(logits.dtype)
