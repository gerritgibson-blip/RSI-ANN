"""Plot the self-improvement curve and final generalisation from log.jsonl."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = os.path.dirname(os.path.abspath(__file__))
recs = [json.loads(l) for l in open(os.path.join(D, "log.jsonl"))]
rounds = [r for r in recs if r["phase"] == "round"]
finals = [r for r in recs if r["phase"] == "final" and "eval" in r]
final = finals[-1] if finals else None
seed = {"add": 5, "mul": 4, "sort": 5, "max": 5}
if final:
    seed = final["seed_levels"]

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for t in seed:
    xs = [r["t"] / 60 for r in rounds]
    ys = [r["frontier"][t] for r in rounds]
    ax[0].step(xs, ys, where="post", label=t, lw=2)
    ax[0].axhline(seed[t], ls=":", lw=0.8, color=ax[0].lines[-1].get_color())
grow_t = [r["t"] / 60 for r in rounds if "grew_to_layers" in r]
for g in grow_t:
    ax[0].axvline(g, color="grey", alpha=0.4)
ax[0].set(xlabel="wall-clock minutes", ylabel="mastered level (>=90% greedy)",
          title="Self-improvement frontier\n(dotted = last level ever given labels; grey = depth growth)")
ax[0].legend()

if final:
    for t, d in final["eval"].items():
        lv = sorted(int(k) for k in d)
        ax[1].plot(lv, [d[str(l)] for l in lv], marker="o", label=t)
        ax[1].axvline(seed[t] + 0.5, ls=":", lw=0.8, color=ax[1].lines[-1].get_color())
    ax[1].set(xlabel="difficulty level (digits)", ylabel="greedy exact-match accuracy",
              title="Final self-test (fresh problems)\nright of dotted line = never labelled")
    ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(D, "progress.png"), dpi=130)
print("saved progress.png")
