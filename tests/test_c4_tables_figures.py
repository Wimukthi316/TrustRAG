"""A missing artefact becomes a visible TODO, not a dropped row."""

from __future__ import annotations

from pathlib import Path


from src.c4_attribution import figures, tables


def test_every_builder_produces_something_even_with_nothing_on_disk(monkeypatch):
    monkeypatch.setattr(tables, "RESULTS", Path("results/c4-does-not-exist"))
    text = tables.build()
    assert text.count("TODO") >= len(tables.BUILDERS)
    for builder in tables.BUILDERS:
        assert builder.__name__.startswith("c4_")


def test_the_table_index_is_contiguous(monkeypatch):
    monkeypatch.setattr(tables, "RESULTS", Path("results/c4-does-not-exist"))
    text = tables.build()
    for number in range(1, len(tables.BUILDERS) + 1):
        assert f"C4-{number} " in text


def test_a_nan_renders_as_na_and_none_renders_as_todo():
    assert tables._f(None) == "TODO"
    assert tables._f(float("nan")) == "n/a"
    assert tables._f(float("inf")) == "inf"
    assert tables._f(0.123456) == "0.1235"


def test_every_figure_name_is_on_the_backends_whitelist():
    from backend.app.services.metrics import FIGURE_ORDER

    for name in figures.FIGURE_NAMES:
        assert name in FIGURE_ORDER


def test_the_figure_builders_and_the_name_list_agree():
    assert tuple(name for name, _ in figures.BUILDERS) == figures.FIGURE_NAMES


def test_a_figure_with_no_artefact_is_skipped_rather_than_drawn_from_nothing(monkeypatch):
    monkeypatch.setattr(figures, "RESULTS", Path("results/c4-does-not-exist"))
    for _, builder in figures.BUILDERS:
        assert builder() is False


def test_arch_diagrams_and_technical_validation_artefacts_exist():
    from src.c4_attribution import arch_diagrams

    for name in arch_diagrams.DIAGRAM_NAMES:
        assert (arch_diagrams.OUT_DIR / f"{name}.png").exists()
        assert (arch_diagrams.OUT_DIR / f"{name}.pdf").exists()
    text = tables.build()
    assert "### C4-15  Runtime latency, throughput, and memory footprint" in text
    assert "### C4-16  User acceptance testing and internal reviewer validation" in text
