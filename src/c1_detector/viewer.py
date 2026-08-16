"""The Localisation Viewer: gold spans and predictions on the same answer.

A table saying span-exact F1 is 0.1485 takes three minutes to explain. A picture
of one answer with the gold span in one colour and the prediction in another
takes three seconds, and the panel sees the contribution before a word is spoken.

Output is a single self-contained HTML file. No server, no build step, no npm:
open it in any browser, email it to anyone, screenshot it for the report. That
matters more than integrating into the React app, because the app has to be
running for anyone to see it and this does not.

Five colours, and every character of the answer gets exactly one:

    green      gold and prediction agree, and the whole span matched exactly
    amber      gold and prediction agree here, but the span boundaries do not
    red        inside a gold span that no prediction covers - the model missed it
    grey       inside a prediction that touches no gold span at all
    outline    inside a prediction that overshoots the gold span it belongs to

Red is the one to look at. The decomposition says the detector under-covers, and
red is what under-coverage looks like: gold text at the edges of a span the model
started too late or ended too early.

Examples are chosen by measurement, not by eye - the same discipline that picked
the demo record. For each failure category the cleanest readable case wins, and
the rule is written down in pick_examples so nobody has to trust the choice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.c1_detector.localisation import (
    classify,
    load_probability_dump,
    overlap_components,
)

CATEGORIES = ("exact", "boundary", "split", "merge", "missed", "spurious")

# Answers shorter than this are usually too thin to show anything; longer ones
# stop fitting on a slide. Both ends are presentation choices, not analysis.
MIN_CHARS = 150
MAX_CHARS = 900
IDEAL_CHARS = 400
PER_CATEGORY = 2


def annotate(record: Dict[str, Any]) -> Dict[str, Any]:
    """Per-character classes plus the span groups behind them."""
    gold = list(record["gold_spans"])
    pred = list(record["pred_spans"])
    answer = record["answer"]
    n = len(answer)

    in_gold = [False] * n
    in_pred = [False] * n
    is_exact = [False] * n
    is_spurious = [False] * n
    groups: List[Dict[str, Any]] = []

    for gold_idx, pred_idx in overlap_components(gold, pred):
        identical = (
            len(gold_idx) == 1
            and len(pred_idx) == 1
            and gold[gold_idx[0]] == pred[pred_idx[0]]
        )
        name = classify(len(gold_idx), len(pred_idx), identical)

        for i in gold_idx:
            start, end = gold[i]
            for k in range(max(0, start), min(n, end)):
                in_gold[k] = True
        for i in pred_idx:
            start, end = pred[i]
            for k in range(max(0, start), min(n, end)):
                in_pred[k] = True
                if name == "exact":
                    is_exact[k] = True
                elif name == "spurious":
                    is_spurious[k] = True

        entry: Dict[str, Any] = {
            "category": name,
            "gold": [list(gold[i]) for i in gold_idx],
            "pred": [list(pred[i]) for i in pred_idx],
            "gold_text": [answer[gold[i][0] : gold[i][1]] for i in gold_idx],
            "pred_text": [answer[pred[i][0] : pred[i][1]] for i in pred_idx],
        }
        if name == "boundary":
            gold_start, gold_end = gold[gold_idx[0]]
            pred_start, pred_end = pred[pred_idx[0]]
            entry["start_delta"] = pred_start - gold_start
            entry["end_delta"] = pred_end - gold_end
        groups.append(entry)

    classes: List[str] = []
    for k in range(n):
        if in_gold[k] and in_pred[k]:
            classes.append("exact" if is_exact[k] else "overlap")
        elif in_gold[k]:
            classes.append("missed")
        elif in_pred[k]:
            classes.append("spurious" if is_spurious[k] else "overreach")
        else:
            classes.append("plain")

    return {"classes": classes, "groups": groups}


def segments(answer: str, classes: Sequence[str]) -> List[Tuple[str, str]]:
    """Collapse the per-character classes into runs, for rendering."""
    runs: List[Tuple[str, str]] = []
    for index, name in enumerate(classes):
        if runs and runs[-1][0] == name:
            runs[-1] = (name, runs[-1][1] + answer[index])
        else:
            runs.append((name, answer[index]))
    return runs


def pick_examples(
    records: Sequence[Dict[str, Any]],
    per_category: int = PER_CATEGORY,
) -> List[Dict[str, Any]]:
    """One or two clean cases per failure category, chosen by a written rule.

    A case qualifies for a category if it contains a group of that category. Of
    those, prefer the answer closest to IDEAL_CHARS and then the fewest groups,
    because a picture with three spans reads and a picture with fifteen does not.
    Ties break on the record id so the choice is reproducible.
    """
    scored: Dict[str, List[Tuple[Any, Dict[str, Any]]]] = {c: [] for c in CATEGORIES}

    for record in records:
        answer = record["answer"]
        if not MIN_CHARS <= len(answer) <= MAX_CHARS:
            continue
        marked = annotate(record)
        present = {group["category"] for group in marked["groups"]}
        key = (abs(len(answer) - IDEAL_CHARS), len(marked["groups"]), str(record["id"]))
        for category in present & set(CATEGORIES):
            scored[category].append((key, {**record, **marked}))

    chosen: List[Dict[str, Any]] = []
    seen: set = set()
    for category in CATEGORIES:
        for _, record in sorted(scored[category], key=lambda pair: pair[0])[:per_category]:
            if record["id"] in seen:
                continue
            seen.add(record["id"])
            chosen.append({**record, "featured": category})
    return chosen


def build_payload(
    records: Sequence[Dict[str, Any]],
    metrics: Dict[str, Any],
    localisation: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Everything the page renders, as plain data so it can be tested."""
    chosen = pick_examples(records)
    overall = metrics["overall"]

    headline = {
        "example_f1": overall["example"]["f1"],
        "span_overlap_f1": overall["span_overlap"]["f1"],
        "span_exact_f1": overall["span_exact"]["f1"],
        "n_examples": overall["n_examples"],
        "n_gold_spans": overall["n_gold_spans"],
        "n_pred_spans": overall["n_pred_spans"],
    }
    if localisation:
        headline["ceiling_span_exact_f1"] = localisation["tokenisation_ceiling"][
            "span_exact"
        ]["f1"]
        headline["buckets"] = localisation["overall"]["buckets"]

    return {
        "headline": headline,
        "examples": [
            {
                "id": str(record["id"]),
                "task_type": record.get("task_type", ""),
                "featured": record["featured"],
                "answer": record["answer"],
                "segments": segments(record["answer"], record["classes"]),
                "groups": record["groups"],
            }
            for record in chosen
        ],
    }


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>TrustRAG C1 - Localisation Viewer</title>
<style>
  :root {
    --ink: #1f2328; --muted: #59636e; --line: #d1d9e0; --bg: #ffffff;
    --exact-bg: #d7f5dd; --exact-ink: #14602c;
    --overlap-bg: #fff2cc; --overlap-ink: #7a5300;
    --missed-bg: #ffdedb; --missed-ink: #a2231c;
    --spurious-bg: #eaeef2; --spurious-ink: #4a545e;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
         color: var(--ink); background: #f6f8fa; }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 28px 24px 64px; }
  h1 { font-size: 20px; margin: 0 0 4px; font-weight: 620; }
  .sub { color: var(--muted); font-size: 13px; margin-bottom: 22px; }
  .strip { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 10px; }
  .stat { background: var(--bg); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; }
  .stat .k { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  .stat .v { font-size: 30px; font-weight: 640; font-variant-numeric: tabular-nums; margin-top: 2px; }
  .stat .n { font-size: 12px; color: var(--muted); margin-top: 4px; }
  .stat.head .v { color: #14602c; }
  .stat.tail .v { color: #a2231c; }
  .caveat { background: #fff8e6; border: 1px solid #e8d38a; border-radius: 10px;
            padding: 11px 14px; font-size: 13px; color: #6a4c00; margin-bottom: 22px; }
  .cols { display: grid; grid-template-columns: 220px 1fr; gap: 20px; align-items: start; }
  .picker { background: var(--bg); border: 1px solid var(--line); border-radius: 10px; padding: 10px; position: sticky; top: 16px; }
  .picker h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .04em;
               color: var(--muted); margin: 6px 6px 8px; font-weight: 600; }
  button.ex { display: block; width: 100%; text-align: left; border: 0; background: transparent;
              padding: 7px 9px; border-radius: 7px; cursor: pointer; font: inherit; font-size: 13px; color: var(--ink); }
  button.ex:hover { background: #eef2f6; }
  button.ex[aria-current="true"] { background: #1f2328; color: #fff; }
  button.ex small { display: block; color: var(--muted); font-size: 11px; }
  button.ex[aria-current="true"] small { color: #c9d1d9; }
  .panel { background: var(--bg); border: 1px solid var(--line); border-radius: 10px; padding: 20px 22px; }
  .answer { font-size: 16px; line-height: 2.0; white-space: pre-wrap; word-wrap: break-word; }
  .answer span { padding: 2px 0; border-radius: 3px; }
  .c-exact { background: var(--exact-bg); color: var(--exact-ink); box-shadow: 0 1px 0 #7bc596 inset, 0 -2px 0 #2da44e inset; }
  .c-overlap { background: var(--overlap-bg); color: var(--overlap-ink); box-shadow: 0 -2px 0 #d4a72c inset; }
  .c-missed { background: var(--missed-bg); color: var(--missed-ink); box-shadow: 0 -2px 0 #cf222e inset; }
  .c-spurious { background: var(--spurious-bg); color: var(--spurious-ink); box-shadow: 0 -2px 0 #8c959f inset; }
  .c-overreach { background: transparent; box-shadow: 0 -2px 0 #d4a72c inset; opacity: .8; }
  .legend { display: flex; flex-wrap: wrap; gap: 8px; margin: 18px 0 0; font-size: 12.5px; }
  .legend b { font-weight: 600; }
  .key { border-radius: 6px; padding: 4px 9px; }
  table { width: 100%; border-collapse: collapse; margin-top: 18px; font-size: 13px; }
  th, td { text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
  th { color: var(--muted); font-weight: 600; font-size: 11.5px; text-transform: uppercase; letter-spacing: .04em; }
  td.q { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }
  .tag { display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: 11.5px; font-weight: 600; }
  .t-exact { background: var(--exact-bg); color: var(--exact-ink); }
  .t-boundary, .t-split, .t-merge { background: var(--overlap-bg); color: var(--overlap-ink); }
  .t-missed { background: var(--missed-bg); color: var(--missed-ink); }
  .t-spurious { background: var(--spurious-bg); color: var(--spurious-ink); }
  .meta { color: var(--muted); font-size: 12.5px; margin-bottom: 14px; }
</style>
<div class="wrap">
  <h1>TrustRAG C1 &mdash; Localisation Viewer</h1>
  <div class="sub">Human-annotated spans against the detector's predictions, on the RAGTruth test split.</div>
  <div class="strip" id="strip"></div>
  <div class="caveat" id="caveat"></div>
  <div class="cols">
    <div class="picker" id="picker"></div>
    <div class="panel">
      <div class="meta" id="meta"></div>
      <div class="answer" id="answer"></div>
      <div class="legend">
        <span class="key c-exact"><b>green</b> boundaries exactly right</span>
        <span class="key c-overlap"><b>amber</b> overlaps gold, boundaries wrong</span>
        <span class="key c-missed"><b>red</b> gold text no prediction covers</span>
        <span class="key c-spurious"><b>grey</b> prediction touching no gold span</span>
        <span class="key c-overreach"><b>underline only</b> prediction overshooting its gold span</span>
      </div>
      <table id="groups"></table>
    </div>
  </div>
</div>
<script id="payload" type="application/json">__PAYLOAD__</script>
<script>
const DATA = JSON.parse(document.getElementById("payload").textContent);
const H = DATA.headline;
const f = (x) => x.toFixed(4);

document.getElementById("strip").innerHTML = `
  <div class="stat head"><div class="k">Response level F1</div><div class="v">${f(H.example_f1)}</div>
    <div class="n">does the answer contain a hallucination at all</div></div>
  <div class="stat"><div class="k">Span F1, overlapping</div><div class="v">${f(H.span_overlap_f1)}</div>
    <div class="n">the prediction touches the right region</div></div>
  <div class="stat tail"><div class="k">Span F1, exact offsets</div><div class="v">${f(H.span_exact_f1)}</div>
    <div class="n">both boundaries character-for-character right</div></div>`;

const ceiling = H.ceiling_span_exact_f1;
document.getElementById("caveat").innerHTML =
  `Read ${f(H.span_exact_f1)} against a <b>measured ceiling of ${ceiling ? f(ceiling) : "n/a"}</b> on this split, not against 1.0 &mdash; `
  + `and against RAGTruth's own report that two human annotators agree on span boundaries only <b>78.8%</b> of the time. `
  + `${H.n_examples.toLocaleString()} responses, ${H.n_gold_spans.toLocaleString()} gold spans, ${H.n_pred_spans.toLocaleString()} predicted spans.`;

const picker = document.getElementById("picker");
let html = '<h2>Examples</h2>';
DATA.examples.forEach((ex, i) => {
  html += `<button class="ex" data-i="${i}">${ex.featured}<small>record ${ex.id} &middot; ${ex.task_type}</small></button>`;
});
picker.innerHTML = html;

function render(i) {
  const ex = DATA.examples[i];
  document.querySelectorAll("button.ex").forEach(b =>
    b.setAttribute("aria-current", String(Number(b.dataset.i) === i)));

  document.getElementById("meta").textContent =
    `record ${ex.id} · ${ex.task_type} · featured category: ${ex.featured}`;

  const box = document.getElementById("answer");
  box.textContent = "";
  ex.segments.forEach(([cls, text]) => {
    const el = document.createElement("span");
    el.className = "c-" + cls;
    el.textContent = text;
    box.appendChild(el);
  });

  let rows = "<tr><th>category</th><th>gold</th><th>prediction</th><th>edges</th></tr>";
  ex.groups.forEach(g => {
    const delta = (g.start_delta === undefined) ? "" :
      `start ${g.start_delta >= 0 ? "+" : ""}${g.start_delta}, end ${g.end_delta >= 0 ? "+" : ""}${g.end_delta}`;
    const esc = (s) => s.replace(/[&<>]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
    rows += `<tr><td><span class="tag t-${g.category}">${g.category}</span></td>`
         +  `<td class="q">${g.gold_text.map(esc).join(" &nbsp;/&nbsp; ") || "&mdash;"}</td>`
         +  `<td class="q">${g.pred_text.map(esc).join(" &nbsp;/&nbsp; ") || "&mdash;"}</td>`
         +  `<td class="q">${delta}</td></tr>`;
  });
  document.getElementById("groups").innerHTML = rows;
}

picker.addEventListener("click", (e) => {
  const b = e.target.closest("button.ex");
  if (b) render(Number(b.dataset.i));
});
render(0);
</script>
"""


def render_page(payload: Dict[str, Any]) -> str:
    """Inline the payload. The closing-tag escape keeps it inside the script tag."""
    blob = json.dumps(payload).replace("</", "<\\/")
    return PAGE.replace("__PAYLOAD__", blob)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the standalone Localisation Viewer page."
    )
    parser.add_argument("--probs", default="results/c1/test/probabilities.jsonl")
    parser.add_argument("--metrics", default="results/c1/test/metrics.json")
    parser.add_argument(
        "--localisation", default="results/c1/analysis/localisation_report.json"
    )
    parser.add_argument("--out", default="results/c1/analysis/localisation_viewer.html")
    args = parser.parse_args(argv)

    records = load_probability_dump(args.probs)
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    localisation_path = Path(args.localisation)
    localisation = (
        json.loads(localisation_path.read_text(encoding="utf-8"))
        if localisation_path.exists()
        else None
    )

    payload = build_payload(records, metrics, localisation)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(payload), encoding="utf-8")

    print(f"{len(payload['examples'])} examples picked from {len(records):,} responses")
    for example in payload["examples"]:
        counts: Dict[str, int] = {}
        for group in example["groups"]:
            counts[group["category"]] = counts.get(group["category"], 0) + 1
        print(
            f"  {example['featured']:<10} record {example['id']:<8} "
            f"{example['task_type']:<14} {counts}"
        )
    print(f"\nwritten: {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
