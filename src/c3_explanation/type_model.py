"""Layer 3.2 of C3: the two-headed transformer that names the error type.

The input is the answer with the span in question wrapped in plain-text markers,
then the question, then the context:

    <<the span>> ... rest of the answer   [SEP]   question   [SEP]   context

The marked answer goes first so that truncation can only cut the far end of the
context and never the marked span. Putting the context first, as C1 does, would
silently produce training examples with nothing marked in them on the long
data-to-text records.

Two classification heads sit on the pooled representation, one per axis:

    mechanism   conflict  the span contradicts something in the context
                baseless  the context says nothing about it
    severity    evident   obvious once you look
                subtle    a small shift in meaning

Two heads rather than one 4-way head is the design decision, not an
implementation detail. The axes are different kinds of question, they are not
equally learnable, and the CPU baseline in type_baseline.py already measured that
gap. A flat 4-way accuracy would average the two together and hide it.

The 4-class distribution Layer 4 needs is the product of the two head marginals.
The baseline fits a flat 4-way model alongside precisely so the cost of that
independence assumption is measured rather than assumed; the same comparison is
available here through evaluate_c3_type.py.

Nothing in this module touches C1's checkpoint, C1's splits or anything under
results/c1/. It reads them; it never writes them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset

from src.c3_explanation.labels_audit import (
    MECHANISM_BASELESS,
    MECHANISM_CONFLICT,
    SEVERITY_EVIDENT,
    SEVERITY_SUBTLE,
    axes_of_error_type,
)
from src.common.schema import ErrorType

MECHANISMS: Tuple[str, ...] = (MECHANISM_BASELESS, MECHANISM_CONFLICT)
SEVERITIES: Tuple[str, ...] = (SEVERITY_EVIDENT, SEVERITY_SUBTLE)
ERROR_TYPES: Tuple[str, ...] = (
    ErrorType.EVIDENT_CONFLICT.value,
    ErrorType.SUBTLE_CONFLICT.value,
    ErrorType.EVIDENT_BASELESS.value,
    ErrorType.SUBTLE_BASELESS.value,
)

SPAN_OPEN = "<<"
SPAN_CLOSE = ">>"


def mark_span(answer: str, start: int, end: int, open_marker: str = SPAN_OPEN,
              close_marker: str = SPAN_CLOSE) -> str:
    """The answer with one span wrapped in markers.

    Offsets are clamped rather than trusted: a span that ran past the end of the
    answer would otherwise produce a silently unmarked example, which is exactly
    the failure this file is arranged to avoid.
    """
    n = len(answer)
    a = max(0, min(int(start), n))
    b = max(a, min(int(end), n))
    return f"{answer[:a]}{open_marker}{answer[a:b]}{close_marker}{answer[b:]}"


def build_input(
    answer: str,
    start: int,
    end: int,
    question: str,
    context: str,
    open_marker: str = SPAN_OPEN,
    close_marker: str = SPAN_CLOSE,
) -> Tuple[str, str]:
    """The (first, second) segment pair handed to the tokenizer.

    The tokenizer truncates the longer segment first with truncation='longest_first',
    and the context is by far the longer one, so the marked answer survives.
    """
    marked = mark_span(answer, start, end, open_marker, close_marker)
    first = marked if not question else f"{marked} [Q] {question}"
    return (first, context)


class TypeExample(Dict[str, Any]):
    """One span: the text pair, the two axis labels, and where it came from."""


def examples_from_rows(
    rows: Sequence[Dict[str, Any]],
    open_marker: str = SPAN_OPEN,
    close_marker: str = SPAN_CLOSE,
) -> List[TypeExample]:
    """Turn feature rows from type_baseline into model inputs.

    The rows carry the span offsets, the response id and the gold or inherited
    error type; the answer, question and context come with them so that this
    module never has to reopen the corpus.
    """
    out: List[TypeExample] = []
    for row in rows:
        error_type = str(row["error_type"])
        mechanism, severity = axes_of_error_type(ErrorType(error_type))
        first, second = build_input(
            row.get("answer", ""),
            int(row["start"]),
            int(row["end"]),
            row.get("question", ""),
            row.get("context", ""),
            open_marker,
            close_marker,
        )
        out.append(
            TypeExample(
                response_id=str(row["response_id"]),
                task_type=str(row.get("task_type", "other")),
                model=str(row.get("model", "unknown")),
                start=int(row["start"]),
                end=int(row["end"]),
                source=str(row.get("source", "gold")),
                error_type=error_type,
                mechanism_label=MECHANISMS.index(str(mechanism)),
                severity_label=SEVERITIES.index(str(severity)),
                first=first,
                second=second,
            )
        )
    return out


class TypeDataset(Dataset):
    """Tokenises on demand so a long corpus does not have to fit in memory twice."""

    def __init__(
        self,
        examples: Sequence[TypeExample],
        tokenizer: Any,
        max_length: int = 1024,
    ) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self.examples[index]
        encoded = self.tokenizer(
            example["first"],
            example["second"],
            truncation="longest_first",
            max_length=self.max_length,
            padding=False,
        )
        encoded["mechanism_label"] = example["mechanism_label"]
        encoded["severity_label"] = example["severity_label"]
        return encoded


def collate(batch: Sequence[Dict[str, Any]], tokenizer: Any) -> Dict[str, torch.Tensor]:
    mechanism = torch.tensor([b.pop("mechanism_label") for b in batch], dtype=torch.long)
    severity = torch.tensor([b.pop("severity_label") for b in batch], dtype=torch.long)
    padded = tokenizer.pad(list(batch), return_tensors="pt")
    padded["mechanism_label"] = mechanism
    padded["severity_label"] = severity
    return padded


class TwoAxisTypeModel(nn.Module):
    """An encoder with one linear head per axis.

    Pooling is a mask-aware mean over the final hidden states rather than a CLS
    token, because ModernBERT has no pretrained pooler and a randomly initialised
    one would add variance for nothing.
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int,
        dropout: float = 0.1,
        n_mechanism: int = len(MECHANISMS),
        n_severity: int = len(SEVERITIES),
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.mechanism_head = nn.Linear(hidden_size, n_mechanism)
        self.severity_head = nn.Linear(hidden_size, n_severity)

    def pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mechanism_label: Optional[torch.Tensor] = None,
        severity_label: Optional[torch.Tensor] = None,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
        **unused: Any,
    ) -> Dict[str, torch.Tensor]:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(self.pool(output.last_hidden_state, attention_mask))
        logits = {
            "mechanism": self.mechanism_head(pooled),
            "severity": self.severity_head(pooled),
        }
        out: Dict[str, torch.Tensor] = dict(logits)
        if mechanism_label is not None and severity_label is not None:
            weights = class_weights or {}
            mechanism_loss = nn.functional.cross_entropy(
                logits["mechanism"], mechanism_label, weight=weights.get("mechanism")
            )
            severity_loss = nn.functional.cross_entropy(
                logits["severity"], severity_label, weight=weights.get("severity")
            )
            out["mechanism_loss"] = mechanism_loss
            out["severity_loss"] = severity_loss
            out["loss"] = mechanism_loss + severity_loss
        return out


def axis_probabilities(logits: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        "mechanism": torch.softmax(logits["mechanism"], dim=-1),
        "severity": torch.softmax(logits["severity"], dim=-1),
    }


def error_type_probabilities(
    mechanism_probs: torch.Tensor, severity_probs: torch.Tensor
) -> torch.Tensor:
    """P(type) = P(mechanism) x P(severity), in ERROR_TYPES order.

    The independence assumption is stated here and measured in
    evaluate_c3_type.py against the same 4-way distribution the CPU baseline's
    flat model produces.
    """
    columns = []
    for error_type in ERROR_TYPES:
        mechanism, severity = axes_of_error_type(ErrorType(error_type))
        columns.append(
            mechanism_probs[:, MECHANISMS.index(str(mechanism))]
            * severity_probs[:, SEVERITIES.index(str(severity))]
        )
    stacked = torch.stack(columns, dim=1)
    return stacked / stacked.sum(dim=1, keepdim=True).clamp(min=1e-12)
