# RSI-ANN: a self-testing, self-training, self-improving neural network

A small decoder-only Transformer that **teaches itself** four automatically checkable algorithmic
tasks, on 4 CPU cores with no GPU:
- multi-digit addition
- n-digit × 1-digit multiplication
- digit sorting
- prefix maximum

It was given correct answers only for problems up to **5 digits (4 for multiplication)**. Above
that, every level was learned from the model's *own* answers: it samples answers, keeps the ones
an exact verifier confirms, and retrains on them. It also grows its own depth when it stalls,
and manages its own memory use under a hard 11 GB cap.

## Current state (round 292, still training)

| Task | Labelled up to | Mastered by self-training (≥90% greedy on fresh problems) |
|---|---|---|
| Addition | 5 | **104 digits** |
| Multiplication (× 1 digit) | 4 | **111 digits** |
| Sorting | 5 | **160 digits** (curriculum cap) |
| Prefix max | 5 | **160 digits** (curriculum cap) |

- Model: 14 layers, d_model 128, **2.54M parameters**. It started at 5 layers and 0.9M
  parameters and grew itself 9 times.
- Weights: `model_weights.pt`, loaded with `GPT(Config(**ckpt["cfg"])).load_state_dict(ckpt["state"])`.
- `progress.png`: frontier over time and the final self-test.
  `figures/progress_round77.png` is an earlier snapshot.

## How it works (`selfimprove.py`)

1. **Hardware probe.** Times training steps at several widths and picks the largest that fits
   the time budget (4 cores → d_model 128).
2. **Self-tested architecture search.** Tries {NoPE, RoPE} × {3, 5 layers}, each scored on
   *harder, never-seen* levels. NoPE won: the RoPE models scored 0% on harder levels for
   addition, sorting and prefix max. NoPE also means there is no fixed context limit.
3. **Seed training** until the model passes its own tests on every seed level (≥97%).
4. **Expert iteration.** Each round, for every task, at frontier+1:
   - Generate fresh problems.
   - Make k attempts per problem (greedy plus sampled).
   - An exact verifier keeps only correct answers.
   - Retrain on the model's own verified answers plus replay of earlier levels, with per-round
     warmup + cosine decay.
   - Self-test on fresh problems; promote the frontier at ≥90%.
5. **Self-improvement when stuck:**
   - Raise sampling effort k (up to 32) when a task stalls.
   - Grow a layer when any task stalls for 6 rounds or every task stalls for 4. Growth is
     function-preserving: the new block starts as a pass-through. The depth limit is 16 layers.
   - Stop on real limits only: all tasks at the level cap, saturation at maximum depth or at
     the memory limit, or the hardware memory limit.
6. **Memory management under a hard 11 GB cap** (memory cgroup, set up by `drive.sh`).
   - A memory-efficiency ladder, from least to most memory saving:
     - recompute 0, 1/3, 1/2 or all layers in the backward pass (activation checkpointing);
     - then gradient accumulation (2× or 4× micro-batches);
     - then a smaller generation batch.
   - Each round's peak memory is logged. Above 85% of the cap the ladder steps up.
   - Below 60% it steps down to save compute, but only if the lower level's *measured* peak,
     scaled to the current depth, would fit.
   - If the cap kills the process, the driver raises the level, records that the level failed,
     and resumes.
   - Every level is verified to give the same gradients as no recompute (relative difference
     0.0 for recompute, about 1e-7 for accumulation).
   - Result: at 13 layers peak memory fell from 10.3 GB to about 6 GB.
7. **Robustness.** Checkpoints every round, including the replay buffer and the stall and
   memory counters (`counters.json`). `drive.sh` chains 110-minute runs until a non-time stop,
   and resuming survived several container restarts.

Model (`model.py`): RMSNorm pre-norm, SwiGLU, grouped-query attention with QK-norm, optional
RoPE (NoPE used), tied embeddings, KV-cache decoding (10× faster generation, identical outputs),
optional per-block activation checkpointing.

## Run log highlights (see `log.jsonl`, `drive.log`)
- **Rounds 1–198:** labels up to 4–5 digits, then all four tasks self-trained to the first
  80-digit cap.
- **Instability fix:** constant-LR training after growth made accuracy swing between rounds
  (sort 0.86 → 0.06). A per-round warmup + cosine schedule fixed it.
- **Stall-counter bug:** counters were not saved in checkpoints, so growth could never trigger
  across 2-hour runs. Fixed by persisting them; growth to 11+ layers followed.
- **Cap raised from 80 to 160** (a curriculum cap, not a model limit). Sorting and max reached
  160 at round 278.
- **11 GB cap and efficiency ladder added at round 281.**

## Files
| File | Purpose |
|---|---|
| `selfimprove.py` | The whole loop: probe, search, seed, expert iteration, growth, memory ladder |
| `model.py` | Transformer and generation, growth, checkpointing |
| `tasks.py` | Task generators and exact verifiers |
| `drive.sh` | Memory-capped driver: chains resumes, handles memory kills |
| `report.py` | Regenerates `progress.png` from `log.jsonl` |
| `log.jsonl` | Every probe, search, round, growth, memory and final-evaluation record |
| `counters.json` | Stall, sampling and memory-ladder state for resuming |
| `model_weights.pt` | Current weights (round 292) |

Full checkpoints (`ckpt*.pt`, which include the 100+ MB self-generated replay buffer) are not
tracked. To resume from the weights alone, `cp model_weights.pt ckpt.pt` and run `./drive.sh`:
the missing replay buffer is regenerated by the model itself (solve, then verify).

## Run
```bash
python3 selfimprove.py --budget_min 105 --seed_steps 12000   # fresh run
./drive.sh                                                   # resume under the 11 GB cap until a real limit
python3 report.py                                            # plot
```

## Scope
This is a genuine self-improving system, but a narrow one. The self-testing relies on exact
verifiers, so it works for checkable tasks like these. The model is about 2.5M parameters. It
is not a general intelligence.
