"""results/cueworld/*.json を1行ずつの表にする。

python -m active_inference.experiments.summarize_cueworld [パターン]
パターンはファイル名の部分一致（例: comm_short）。省略すると全部。
"""

from __future__ import annotations

import json
import sys

from active_inference.experiments.cueworld_subtasks import RESULTS_DIR


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(p for p in RESULTS_DIR.glob("*.json") if pattern in p.name)
    if not files:
        print(f"{RESULTS_DIR} に該当する結果がありません。")
        return
    print(f"{'name':60s} {'det(agent)':>10s} {'det(rand)':>10s} {'return':>8s}  action_hist(agent)")
    for p in files:
        r = json.loads(p.read_text())
        a, b = r["agent"], r["random"]
        hist = " ".join(f"{x:.2f}" for x in a["action_hist"])
        print(
            f"{r['name']:60s} {a['frac_determined']:10.3f} {b['frac_determined']:10.3f}"
            f" {a['mean_return']:8.3f}  [{hist}]"
        )


if __name__ == "__main__":
    main()
