"""
Aggregate kv_ablation/* files and plot:

 - X-axis: ablation type  (first, second, third, all, cot)
 - clusters: hint type
 - bar pairs:   verbalized vs non-verbalized
 - hue-rows:    ≥.3 prob by 10 % / 40 % / 80 % reasoning-progress

Writes an interactive HTML (plotly) and also returns a pandas DataFrame
ready to `display()`.
"""

from __future__ import annotations
import os, json
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt

def _read_json(p):
    with open(p) as f: return json.load(f)

def load_ablation_dataframe(dataset: str, model_name: str,
                            hint_types: List[str],
                            n_questions: int) -> pd.DataFrame:
    rows = []
    for hint in hint_types:
        base = f"data/{dataset}/{model_name}/{hint}"
        # map qid → verbalization, logprobs stats
        verif = {d["question_id"]: d["verbalizes_hint"]
                 for d in _read_json(f"{base}/hint_verification_with_{n_questions}.json")}
        logprobs = _read_json(f"data/mmlu/DeepSeek-R1-Distill-Llama-8B/logprobs_analysis/{hint}_logprobs.json")["results"]
        def _passed(qid, pct):
            try:
                # ≥0.3 prob on correct option by pct%
                correct = max(logprobs[str(qid)]["dots"],
                              logprobs[str(qid)]["dots_eot"],
                              key=lambda x: x["token_index"])["logprobs"]
                ok = max(correct.values())/sum(correct.values()) >= .3
                return ok
            except Exception:
                return False

        kv_dir = f"{base}/kv_ablation"
        merged_fname = f"kv_ablation_results_with_{n_questions}.json"
        merged_path  = os.path.join(kv_dir, merged_fname)

        if os.path.exists(merged_path):
            # ← new layout: one file → list[dict]
            records = _read_json(merged_path)
            file_records = records if isinstance(records, list) else [records]
        else:
            # ← fallback: many small files
            file_records = []
            for fname in os.listdir(kv_dir):
                rec = _read_json(os.path.join(kv_dir, fname))
                file_records.extend(rec if isinstance(rec, list) else [rec])

        for rec in file_records:
            qid = rec["question_id"]
            rows.append(dict(
                hint_type=hint,
                experiment=rec["experiment"],
                changed=rec["changed"],
                verbalized=verif.get(qid, False),
                early10=_passed(qid, 10),
                early40=_passed(qid, 40),
                early80=_passed(qid, 80),
            ))

    return pd.DataFrame(rows)


def plot(df: pd.DataFrame):
    """
    Make bar chart of % answers that changed after ablation.
    Returns (fig, aggregated_dataframe).
    """
    # aggregate: mean over "changed" for every slice we care about
    key_cols = ["hint_type", "experiment", "verbalized",
                "early10", "early40", "early80"]
    agg = (
        df.groupby(key_cols, observed=True)["changed"]
          .mean()
          .reset_index()
    )

    # one bar per row in agg; position them manually
    fig, ax = plt.subplots(figsize=(14, 6))
    bar_positions = range(len(agg))
    heights = agg["changed"] * 100
    ax.bar(bar_positions, heights)

    # add % labels
    for x, h in zip(bar_positions, heights):
        ax.text(x, h + 1, f"{h:.1f}%", ha="center", fontsize=8)

    # x-tick labels: concatenate the key columns
    labels = (
        agg[key_cols]
        .astype(str)
        .agg(" | ".join, axis=1)
        .str.replace("True", "✓")
        .str.replace("False", "✗")
    )
    ax.set_xticks(bar_positions, labels, rotation=90)
    ax.set_ylabel("% answers changed")
    ax.set_title("KV-ablation impact by condition")
    plt.tight_layout()
    return fig, agg

