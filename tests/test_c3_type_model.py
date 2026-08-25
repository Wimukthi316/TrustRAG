"""Tests for the C3 two-axis type head.

No network and no pretrained download: the encoder is replaced by a stub so the
suite stays runnable offline. What is tested is the part that would silently
produce a broken training set -- the span markers, the segment order that keeps
them alive under truncation, and the product that turns two axis marginals into
the four-class distribution Layer 4 consumes.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.c3_explanation.type_model import (
    ERROR_TYPES,
    MECHANISMS,
    SEVERITIES,
    TwoAxisTypeModel,
    build_input,
    error_type_probabilities,
    examples_from_rows,
    mark_span,
)


class StubEncoder(nn.Module):
    """Returns a fixed per-token embedding so pooling can be checked by hand."""

    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.hidden = hidden
        self.embedding = nn.Embedding(50, hidden)

    def forward(self, input_ids, attention_mask=None, **unused):
        class Output:
            pass

        out = Output()
        out.last_hidden_state = self.embedding(input_ids)
        return out


def test_mark_span_wraps_exactly_the_span():
    assert mark_span("abcdef", 2, 4) == "ab<<cd>>ef"
    assert mark_span("abcdef", 0, 6) == "<<abcdef>>"


def test_mark_span_clamps_offsets_rather_than_dropping_the_markers():
    """An out-of-range span must not produce a silently unmarked example."""
    assert mark_span("abc", 1, 99) == "a<<bc>>"
    assert mark_span("abc", -5, 2) == "<<ab>>c"
    marked = mark_span("abc", 5, 9)
    assert "<<" in marked and ">>" in marked


def test_the_marked_answer_is_the_first_segment():
    """Truncation cuts the longer segment first, and the context is the longer one.

    Putting the context first, as C1 does, would drop the marked span on the long
    data-to-text records and train the model on inputs with nothing marked.
    """
    first, second = build_input(
        "the answer text", 4, 10, "the question", "a very long context " * 50
    )
    assert first.startswith("the <<answer>> text")
    assert "[Q] the question" in first
    assert second.startswith("a very long context")
    assert len(second) > len(first)


def test_build_input_without_a_question_omits_the_marker():
    first, _ = build_input("abcdef", 2, 4, "", "context")
    assert first == "ab<<cd>>ef"
    assert "[Q]" not in first


def test_examples_carry_both_axis_labels():
    rows = [
        {
            "response_id": "r1",
            "start": 0,
            "end": 3,
            "error_type": "subtle_conflict",
            "answer": "abcdef",
            "question": "q",
            "context": "c",
            "task_type": "qa",
        }
    ]
    example = examples_from_rows(rows)[0]
    assert MECHANISMS[example["mechanism_label"]] == "conflict"
    assert SEVERITIES[example["severity_label"]] == "subtle"
    assert example["first"].startswith("<<abc>>")


def test_error_type_probabilities_is_the_outer_product():
    mechanism = torch.tensor([[0.25, 0.75]])  # baseless, conflict
    severity = torch.tensor([[0.4, 0.6]])  # evident, subtle
    out = error_type_probabilities(mechanism, severity)
    assert float(out.sum()) == pytest.approx(1.0)
    expected = {
        "evident_conflict": 0.75 * 0.4,
        "subtle_conflict": 0.75 * 0.6,
        "evident_baseless_info": 0.25 * 0.4,
        "subtle_baseless_info": 0.25 * 0.6,
    }
    for column, name in enumerate(ERROR_TYPES):
        assert float(out[0, column]) == pytest.approx(expected[name])


def test_pooling_ignores_padding():
    """A padded batch and an unpadded one must pool to the same vector."""
    model = TwoAxisTypeModel(StubEncoder(), hidden_size=4, dropout=0.0)
    hidden = torch.tensor(
        [[[1.0, 1.0, 1.0, 1.0], [3.0, 3.0, 3.0, 3.0], [99.0, 99.0, 99.0, 99.0]]]
    )
    mask = torch.tensor([[1, 1, 0]])
    pooled = model.pool(hidden, mask)
    assert torch.allclose(pooled, torch.full((1, 4), 2.0))


def test_forward_returns_two_losses_that_sum_to_the_total():
    torch.manual_seed(0)
    model = TwoAxisTypeModel(StubEncoder(), hidden_size=4, dropout=0.0)
    out = model(
        input_ids=torch.tensor([[1, 2, 3], [4, 5, 6]]),
        attention_mask=torch.tensor([[1, 1, 1], [1, 1, 0]]),
        mechanism_label=torch.tensor([0, 1]),
        severity_label=torch.tensor([1, 0]),
    )
    assert out["mechanism"].shape == (2, len(MECHANISMS))
    assert out["severity"].shape == (2, len(SEVERITIES))
    assert float(out["loss"].detach()) == pytest.approx(
        float(out["mechanism_loss"].detach())
        + float(out["severity_loss"].detach()),
        rel=1e-6,
    )


def test_forward_without_labels_returns_logits_only():
    model = TwoAxisTypeModel(StubEncoder(), hidden_size=4, dropout=0.0)
    out = model(
        input_ids=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
    )
    assert "loss" not in out
    assert set(out) == {"mechanism", "severity"}


def test_resolve_max_length_clamps_to_the_encoders_positions():
    from src.c3_explanation.train_c3_type import resolve_max_length

    class Small:
        max_position_embeddings = 512

    class Large:
        max_position_embeddings = 8192

    assert resolve_max_length(1024, Small()) == 512
    assert resolve_max_length(1024, Large()) == 1024
    assert resolve_max_length(1024, object()) == 1024
