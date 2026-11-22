"""Procedurally generated, automatically verifiable tasks with difficulty levels.

Every task exposes:
  sample(level, rng) -> (prompt_str, answer_str)
  verify(prompt_str, answer_str) -> bool   (independent checker)
Numbers are written least-significant-digit first (reversed), which is known
to make carry-propagation learnable left-to-right.
"""
import random

VOCAB = list("0123456789+*=SRM:,") + ["<pad>", "<eos>"]
STOI = {s: i for i, s in enumerate(VOCAB)}
PAD, EOS = STOI["<pad>"], STOI["<eos>"]


def encode(s):
    return [STOI[ch] for ch in s]


def decode(ids):
    out = []
    for i in ids:
        if i == EOS or i == PAD:
            break
        out.append(VOCAB[i])
    return "".join(out)


def rnum(level, rng):
    """Random number with exactly `level` digits, returned reversed."""
    if level == 1:
        n = rng.randint(0, 9)
    else:
        n = rng.randint(10 ** (level - 1), 10 ** level - 1)
    return str(n)[::-1]


class Add:
    name = "add"

    def sample(self, level, rng):
        a, b = rnum(level, rng), rnum(level, rng)
        return f"{a}+{b}=", self.solve(a, b)

    @staticmethod
    def solve(a, b):
        return str(int(a[::-1]) + int(b[::-1]))[::-1]

    def verify(self, prompt, ans):
        a, b = prompt[:-1].split("+")
        return ans == self.solve(a, b)


class Mul1:
    """Multiply an n-digit number by a 1-digit number."""
    name = "mul"

    def sample(self, level, rng):
        a, b = rnum(level, rng), str(rng.randint(0, 9))
        return f"{a}*{b}=", self.solve(a, b)

    @staticmethod
    def solve(a, b):
        return str(int(a[::-1]) * int(b))[::-1]

    def verify(self, prompt, ans):
        a, b = prompt[:-1].split("*")
        return ans == self.solve(a, b)


class Sort:
    """Sort a string of `level` digits."""
    name = "sort"

    def sample(self, level, rng):
        s = "".join(str(rng.randint(0, 9)) for _ in range(level))
        return f"S:{s}=", "".join(sorted(s))

    def verify(self, prompt, ans):
        return ans == "".join(sorted(prompt[2:-1]))


class Max:
    """Running maximum of a digit sequence (prefix max at each position)."""
    name = "max"

    def sample(self, level, rng):
        s = "".join(str(rng.randint(0, 9)) for _ in range(level))
        return f"M:{s}=", self.solve(s)

    @staticmethod
    def solve(s):
        out, m = [], "0"
        for ch in s:
            m = max(m, ch)
            out.append(m)
        return "".join(out)

    def verify(self, prompt, ans):
        return ans == self.solve(prompt[2:-1])


TASKS = {t.name: t for t in (Add(), Mul1(), Sort(), Max())}
