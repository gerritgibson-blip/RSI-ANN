"""Self-testing / self-training / self-improving loop.

Phase 0  Hardware probe   - measure training throughput to size the model.
Phase 1  Architecture search - train candidate architectures briefly on seed
         levels; score them by the model's own tests, including
         out-of-distribution (harder, never-seen) levels. Keep the best.
Phase 2  Expert iteration - for each task, at the first level above the
         mastered frontier:
           * sample k candidate answers per fresh problem (greedy + temperature)
           * an independent verifier keeps only correct ones
           * retrain on the model's own verified outputs (plus replay)
           * self-test on fresh problems; promote the frontier if acc >= PASS
         When progress stalls, self-improve: raise sampling effort (k), then
         grow the network depth (function-preserving) and keep going.

Ground-truth labels are only ever given for the SEED levels. Every harder level
is reached solely through the model's own verified outputs.
"""
import argparse
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from model import GPT, Config
from tasks import TASKS, VOCAB, PAD, EOS, encode, decode

torch.set_num_threads(os.cpu_count() or 4)
OUT = os.path.dirname(os.path.abspath(__file__))
PASS = 0.90
SEED_LEVELS = {"add": 5, "mul": 4, "sort": 5, "max": 5}
MAX_LEVEL = 160  # NoPE has no context limit; this only bounds the curriculum
MAX_LAYERS = 16
MEM_CAP_GB = 11.0          # hard cap, enforced by a cgroup in drive.sh
MEM_SOFT = 0.85            # climb the efficiency ladder when a round peaks above this fraction
MEM_LOW = 0.60             # step back down (less recompute, faster) when a round peaks below this
# memory-efficiency ladder: (fraction of blocks checkpointed, grad-accumulation micro-batches, generation batch)
MEM_LADDER = [(0.0, 1, 256), (1 / 3, 1, 256), (0.5, 1, 256), (1.0, 1, 256), (1.0, 2, 256), (1.0, 4, 128)]
MEM_LADDER_VERSION = 2
OLD_LADDER_MAP = {0: 0, 1: 3, 2: 4, 3: 5}  # v1 indices -> v2 indices
GEN_BS = 256


def peak_rss_gb():
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1048576
    return 0.0


def reset_peak_rss():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
    except OSError:
        pass


def apply_mem_level(model, level):
    global GEN_BS
    frac, accum, gen_bs = MEM_LADDER[level]
    model.ckpt_frac = frac
    GEN_BS = gen_bs
    return accum
MAX_LEN = 256


def log(rec, path):
    rec["t"] = round(time.time() - T0, 1)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec), flush=True)


# ---------------------------------------------------------------- data
class Buffer:
    """Training pool: infinite generator-labelled seed data + stored self data."""

    def __init__(self, rng, cap_per_level=6000):
        self.rng = rng
        self.cap = cap_per_level
        self.self_data = {t: {} for t in TASKS}  # task -> level -> [(p, a)]
        self.frontier = dict(SEED_LEVELS)

    def add_self(self, task, level, pairs):
        lst = self.self_data[task].setdefault(level, [])
        lst.extend(pairs)
        if len(lst) > self.cap:
            del lst[: len(lst) - self.cap]

    def levels(self, task):
        lv = list(range(1, SEED_LEVELS[task] + 1))
        lv += [l for l, d in self.self_data[task].items() if d]
        return sorted(set(lv))

    def sample(self, n):
        out = []
        tasks = list(TASKS)
        for _ in range(n):
            t = self.rng.choice(tasks)
            lvls = self.levels(t)
            # bias toward the hardest levels (where learning happens)
            w = [1.0 + 2.0 * (i >= len(lvls) - 3) for i in range(len(lvls))]
            lvl = self.rng.choices(lvls, w)[0]
            if lvl <= SEED_LEVELS[t]:
                out.append(TASKS[t].sample(lvl, self.rng))
            else:
                out.append(self.rng.choice(self.self_data[t][lvl]))
        return out


def make_batch(pairs):
    seqs, masks = [], []
    for p, a in pairs:
        pi, ai = encode(p), encode(a) + [EOS]
        seqs.append(pi + ai)
        masks.append([0] * len(pi) + [1] * len(ai))
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), PAD)
    m = torch.zeros((len(seqs), L))
    for i, (s, mk) in enumerate(zip(seqs, masks)):
        x[i, : len(s)] = torch.tensor(s)
        m[i, : len(mk)] = torch.tensor(mk, dtype=torch.float)
    return x, m


def train_steps(model, opt, buf, steps, bs, lr_fn, step0=0, accum=1):
    """accum > 1 splits each batch into micro-batches (same effective batch, less memory)."""
    model.train()
    tot = 0.0
    for s in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_fn(step0 + s)
        opt.zero_grad(set_to_none=True)
        pairs = buf.sample(bs)
        n_tok = sum(len(a) + 1 for _, a in pairs)  # answer tokens + EOS over the full batch
        step_loss = 0.0
        mb = (bs + accum - 1) // accum
        for i in range(0, bs, mb):
            x, m = make_batch(pairs[i:i + mb])
            logits = model(x[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="none")
            loss = (loss * m[:, 1:].reshape(-1)).sum() / n_tok
            loss.backward()
            step_loss += loss.item()
            del logits, loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += step_loss
    return tot / max(steps, 1)


# ---------------------------------------------------------------- self-test
@torch.no_grad()
def attempt(model, task, level, n, rng, k=1, temp=0.8, bs=None):
    """Generate n fresh problems; return (greedy_accuracy, verified_pairs).
    verified_pairs contains at most one verified answer per problem, drawn from
    k attempts (1 greedy + k-1 sampled)."""
    model.eval()
    bs = bs or GEN_BS
    T = TASKS[task]
    probs = [T.sample(level, rng)[0] for _ in range(n)]  # labels discarded
    max_new = level + 4
    greedy_ok, verified = 0, []
    for i in range(0, n, bs):
        chunk = probs[i : i + bs]
        x = torch.tensor([encode(p) for p in chunk])
        solved = [None] * len(chunk)
        for j in range(k):
            ys = model.generate(x, max_new, EOS, PAD, temperature=0.0 if j == 0 else temp)
            for r, y in enumerate(ys.tolist()):
                ans = decode(y)
                ok = T.verify(chunk[r], ans)
                if j == 0 and ok:
                    greedy_ok += 1
                if ok and solved[r] is None:
                    solved[r] = ans
            if all(s is not None for s in solved):
                break
        verified += [(p, a) for p, a in zip(chunk, solved) if a is not None]
    return greedy_ok / n, verified


def full_eval(model, rng, levels, n=200):
    res = {}
    for t, lvls in levels.items():
        res[t] = {l: round(attempt(model, t, l, n, rng)[0], 3) for l in lvls}
    return res


# ---------------------------------------------------------------- phases
def new_model(arch):
    c = Config(vocab=len(VOCAB), max_len=MAX_LEN, **arch)
    return GPT(c)


def make_opt(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.05)


def cosine(lr, warm, total):
    def f(s):
        if s < warm:
            return lr * (s + 1) / warm
        p = min(1.0, (s - warm) / max(1, total - warm))
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))
    return f


def probe(arch, bs, rng):
    m = new_model(arch)
    opt = make_opt(m, 1e-3)
    buf = Buffer(rng)
    train_steps(m, opt, buf, 3, bs, lambda s: 1e-3)
    t = time.time()
    train_steps(m, opt, buf, 10, bs, lambda s: 1e-3)
    return (time.time() - t) / 10, m.n_params()


def arch_search(cands, steps, bs, lr, rng, logp):
    best = None
    for arch in cands:
        m = new_model(arch)
        opt = make_opt(m, lr)
        buf = Buffer(rng)
        loss = train_steps(m, opt, buf, steps, bs, cosine(lr, 100, steps))
        ind = {t: [SEED_LEVELS[t]] for t in TASKS}
        ood = {t: [SEED_LEVELS[t] + 1, SEED_LEVELS[t] + 2] for t in TASKS}
        e_in = full_eval(m, rng, ind, n=128)
        e_ood = full_eval(m, rng, ood, n=128)
        a_in = sum(v for d in e_in.values() for v in d.values()) / len(TASKS)
        a_ood = sum(v for d in e_ood.values() for v in d.values()) / (2 * len(TASKS))
        score = a_in + 2 * a_ood  # generalisation is what powers self-training
        log({"phase": "arch_search", "arch": arch, "params": m.n_params(), "loss": round(loss, 4),
             "in_dist": e_in, "ood": e_ood, "score": round(score, 4)}, logp)
        if best is None or score > best[0]:
            best = (score, arch, m)
    return best


def main():
    global T0
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget_min", type=float, default=100)
    ap.add_argument("--search_steps", type=int, default=1500)
    ap.add_argument("--seed_steps", type=int, default=3000)
    ap.add_argument("--round_steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--n_problems", type=int, default=768)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    T0 = time.time()
    rng = random.Random(args.seed + (1000 if args.resume else 0))
    torch.manual_seed(args.seed)
    logp = os.path.join(OUT, "log.jsonl")
    ckp = os.path.join(OUT, "ckpt.pt")
    if args.resume:
        model, buf, opt, rnd, grows = resume(ckp, rng, args, logp)
        deadline = time.time() + args.budget_min * 60
    else:
        open(logp, "w").close()
        deadline = T0 + args.budget_min * 60
        model, buf, opt = fresh_start(args, rng, logp)
        rnd, grows = 0, 0
    self_improve(model, buf, opt, rnd, grows, args, rng, logp, ckp, deadline)


def resume(ckp, rng, args, logp):
    global T0
    ck = torch.load(ckp, weights_only=False)
    model = GPT(Config(**ck["cfg"]))
    model.load_state_dict(ck["state"])
    buf = Buffer(rng)
    buf.frontier = dict(ck["frontier"])
    T0 = time.time() - ck.get("elapsed", json.loads(open(logp).readlines()[-1])["t"])
    if ck.get("self_data"):
        buf.self_data = ck["self_data"]
        how = "restored"
    else:  # rebuild replay exactly as originally produced: model solves, verifier filters
        for t in TASKS:
            for lvl in range(SEED_LEVELS[t] + 1, buf.frontier[t] + 1):
                _, ver = attempt(model, t, lvl, 768, rng, k=4)
                buf.add_self(t, lvl, ver)
        how = "regenerated_by_model"
    n = sum(len(v) for d in buf.self_data.values() for v in d.values())
    rnd = ck.get("round", max([json.loads(l).get("round", 0) for l in open(logp)]))
    log({"phase": "resume", "frontier": buf.frontier, "replay": how, "replay_examples": n,
         "round": rnd, "params": model.n_params(), "n_layer": model.c.n_layer}, logp)
    return model, buf, make_opt(model, args.lr * 0.5), rnd, ck.get("grows", 0)


def fresh_start(args, rng, logp):

    # Phase 0: hardware probe -> pick the largest width that fits the budget
    sizes = [dict(d_model=128, n_head=4, n_kv_head=2), dict(d_model=192, n_head=6, n_kv_head=2),
             dict(d_model=256, n_head=8, n_kv_head=2), dict(d_model=384, n_head=8, n_kv_head=2)]
    target_step = 0.08 if not args.quick else 0.05  # seconds/step we can afford on this machine
    chosen = sizes[0]
    for s in sizes:
        sec, n = probe(dict(s, n_layer=4, pos="nope"), args.bs, rng)
        log({"phase": "probe", "size": s, "params": n, "sec_per_step": round(sec, 4)}, logp)
        if sec <= target_step:
            chosen = s
    # Phase 1: architecture search over positional scheme x depth
    cands = [dict(chosen, n_layer=L, pos=p) for p in ("nope", "rope") for L in (3, 5)]
    if args.quick:
        cands = cands[:2]
    score, arch, model = arch_search(cands, args.search_steps, args.bs, args.lr, rng, logp)
    log({"phase": "arch_selected", "arch": arch, "score": round(score, 4), "params": model.n_params()}, logp)

    # finish seed training on the chosen architecture
    buf = Buffer(rng)
    opt = make_opt(model, args.lr)
    # self-tested gate: keep training until every seed level is mastered (or cap)
    sched, done, chunk = cosine(args.lr, 50, args.seed_steps), 0, 500
    while done < args.seed_steps:
        loss = train_steps(model, opt, buf, chunk, args.bs, sched, step0=done)
        done += chunk
        ev = full_eval(model, rng, {t: list(range(1, SEED_LEVELS[t] + 1)) for t in TASKS}, n=128)
        worst = min(v for d in ev.values() for v in d.values())
        log({"phase": "seed_train", "steps": done, "loss": round(loss, 4), "worst_seed_acc": worst, "eval": ev}, logp)
        if worst >= 0.97:
            break
    return model, buf, opt


def self_improve(model, buf, opt, rnd, grows, args, rng, logp, ckp, deadline):
    # Phase 2: expert iteration
    k = {t: 4 for t in TASKS}
    stall = {t: 0 for t in TASKS}
    global_stall = 0
    cnt_path = os.path.join(OUT, "counters.json")
    if args.resume and os.path.exists(cnt_path):  # stall state must survive resumes
        cs = json.load(open(cnt_path))
        k, stall, global_stall = cs["k"], cs["stall"], cs["global_stall"]
    mem_level = 0
    if args.resume and os.path.exists(cnt_path):
        cs = json.load(open(cnt_path))
        mem_level = cs.get("mem_level", 0)
        if cs.get("mem_ladder", 1) != MEM_LADDER_VERSION:
            mem_level = OLD_LADDER_MAP.get(mem_level, len(MEM_LADDER) - 1)
    level_peaks = {}  # ladder level -> [observed peak GB, n_layer at the time]
    if args.resume and os.path.exists(cnt_path):
        level_peaks = json.load(open(cnt_path)).get("level_peaks", {})
    accum = apply_mem_level(model, mem_level)
    round_lr = args.lr * 0.5
    stop_reason = "time_budget"
    while time.time() < deadline:
        if all(buf.frontier[t] >= MAX_LEVEL for t in TASKS):
            stop_reason = "all_tasks_at_max_level"
            break
        rnd += 1
        reset_peak_rss()
        promoted_any = False
        summary = {}
        for t in TASKS:
            if buf.frontier[t] >= MAX_LEVEL:
                continue
            lvl = buf.frontier[t] + 1
            acc, ver = attempt(model, t, lvl, args.n_problems, rng, k=k[t])
            buf.add_self(t, lvl, ver)
            summary[t] = {"level": lvl, "greedy_acc": round(acc, 3), "verified": len(ver), "k": k[t]}
        # per-round warmup + cosine decay so every self-test sees a settled model
        loss = train_steps(model, opt, buf, args.round_steps, args.bs,
                           lambda s: round_lr * min(1.0, (s + 1) / 30) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * s / args.round_steps))), accum=accum)
        # self-test after training on fresh problems
        for t in summary:
            lvl = summary[t]["level"]
            acc, _ = attempt(model, t, lvl, 256, rng)
            retain, _ = attempt(model, t, max(1, lvl - 1), 128, rng)
            summary[t].update(post_acc=round(acc, 3), retain_prev=round(retain, 3))
            if acc >= PASS:
                buf.frontier[t] = lvl
                stall[t], k[t] = 0, 4
                promoted_any = True
            else:
                stall[t] += 1
                if stall[t] % 3 == 0:
                    k[t] = min(k[t] * 2, 32)  # self-improve: spend more search effort
        global_stall = 0 if promoted_any else global_stall + 1
        rec = {"phase": "round", "round": rnd, "loss": round(loss, 4), "tasks": summary,
               "frontier": dict(buf.frontier), "params": model.n_params(), "n_layer": model.c.n_layer}
        peak = peak_rss_gb()
        rec.update(peak_rss_gb=round(peak, 2), mem_level=mem_level)
        level_peaks[str(mem_level)] = [round(peak, 3), model.c.n_layer]
        # self-improve memory efficiency: near the cap, climb the ladder
        if peak > MEM_SOFT * MEM_CAP_GB:
            if mem_level + 1 < len(MEM_LADDER):
                mem_level += 1
                accum = apply_mem_level(model, mem_level)
                rec["mem_level_raised_to"] = mem_level
            elif peak > 0.95 * MEM_CAP_GB:
                stop_reason = "hardware_memory_limit"
        elif peak < MEM_LOW * MEM_CAP_GB and mem_level > 0:  # headroom: trade memory back for speed
            # only if the lower level is known (or unmeasured) to fit, scaled to the current depth
            known = level_peaks.get(str(mem_level - 1))
            pred = None if known is None else known[0] * model.c.n_layer / known[1]
            if pred is None or pred < MEM_SOFT * MEM_CAP_GB:
                mem_level -= 1
                accum = apply_mem_level(model, mem_level)
                rec["mem_level_lowered_to"] = mem_level
            else:
                rec["mem_step_down_blocked_pred_gb"] = round(pred, 2)
        # self-improve: grow capacity when everything stalls or any task is stuck
        stuck = global_stall >= 4 or max(stall.values()) >= 6
        mem_ok = not (mem_level + 1 >= len(MEM_LADDER) and peak > MEM_SOFT * MEM_CAP_GB)
        if stuck and model.c.n_layer < MAX_LAYERS and mem_ok:
            model.grow_depth()
            opt = make_opt(model, round_lr)
            grows += 1
            global_stall = 0
            stall = {t: 0 for t in TASKS}
            rec["grew_to_layers"] = model.c.n_layer
        log(rec, logp)
        if model.c.n_layer >= MAX_LAYERS and global_stall >= 8:
            stop_reason = "saturated_at_max_capacity"
        elif not mem_ok and global_stall >= 8:
            stop_reason = "saturated_at_memory_limit"
        torch.save({"cfg": model.c.to_dict(), "state": model.state_dict(), "frontier": buf.frontier,
                    "self_data": buf.self_data, "round": rnd, "grows": grows,
                    "elapsed": time.time() - T0}, ckp + ".tmp")
        os.replace(ckp + ".tmp", ckp)
        json.dump({"k": k, "stall": stall, "global_stall": global_stall, "round": rnd,
                   "mem_level": mem_level, "mem_ladder": MEM_LADDER_VERSION,
                   "level_peaks": level_peaks}, open(cnt_path, "w"))
        if stop_reason != "time_budget":
            break

    # final report: test well beyond the frontier
    if stop_reason == "time_budget":  # intermediate chunk: skip the costly full evaluation
        log({"phase": "final", "frontier": buf.frontier, "stop_reason": stop_reason,
             "params": model.n_params(), "rounds": rnd, "grows": grows}, logp)
        return
    lv = {t: sorted(set(list(range(1, buf.frontier[t] + 4)))) for t in TASKS}
    final = full_eval(model, rng, lv, n=200)
    log({"phase": "final", "frontier": buf.frontier, "seed_levels": SEED_LEVELS, "eval": final,
         "params": model.n_params(), "arch": model.c.to_dict(), "rounds": rnd, "grows": grows,
         "stop_reason": stop_reason}, logp)


if __name__ == "__main__":
    main()
