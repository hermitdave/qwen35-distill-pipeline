# Capacity analysis: why 2B before 4B before 9B

The obvious order is largest student first. The correct order is the reverse.

## The setup

| Role | Model | Total params | Active params |
|---|---|---|---|
| Teacher | Agnes 3 Qwen (MoE) | 125B | **6B** |
| Student 1 | Qwen3.5-9B | 9B | 9B |
| Student 2 | Qwen3.5-4B | 4B | 4B |
| Student 3 | Qwen3.5-2B | 2B | 2B |

## The capacity-gap law

Optimal teacher size scales roughly linearly with the student:
**T\* ≈ 2.5 × S** (Law of Capacity Gap, arXiv 2311.07052). Beyond that ratio the
student does worse than training alone (Distillation Scaling Laws, arXiv
2502.08606).

## The MoE complication

For a **MoE teacher**, the relevant capacity is the **active** parameter count —
the number of parameters the student actually sees per token. A 125B teacher with
6B active is *not* a 125B teacher from the student's point of view.

| Student | Active ratio (teacher/student) | Verdict |
|---|---|---|
| 9B | 6B / 9B = **0.67×** | ❌ Inverted — the student is *larger* than the teacher's active pathway |
| 4B | 6B / 4B = **1.5×** | ✅ Reasonable |
| 2B | 6B / 2B = **3.0×** | ✅ Correct band (≈ 2.5× optimum) |

## What this predicts

* **9B student** — the teacher's active pathway is narrower than the student. The
  student can absorb *style* but not *capability*. Expect logic gains with
  general-ability regressions.
* **4B student** — the teacher is larger; absorption is plausible.
* **2B student** — the correct ratio; the student has the most to gain.

## What was measured

The 2B run confirmed the prediction:

| Category | Base 2B | Distilled 2B |
|---|---|---|
| Logic & Algorithmic | 75 | **88** |
| Subtle & Tricky | 100 | 88 |
| Overall | 91 | 91 |

The 2B student absorbed the teacher's algorithmic behaviour (+13) at the cost of
some edge-case fragility (−12, one task). The 9B Flash-Distilled model from an
earlier run showed the inverted-gap signature instead: logic up, but Syntax &
Surface 67 → 36 and Red Herring 62 → 30.

## The practical rule

**Pilot the smallest student first.** It is the cheapest to run, the most likely
to show a real gain, and it validates the data and config before you spend hours
on the larger students.

If the smallest student shows no gain, the data or config is wrong — and you
have not wasted a 5-hour 9B run finding that out.

## Caveat

The capacity-gap law was formulated for **dense** teachers. For MoE teachers,
neither total nor active parameters reliably predict absorption — a MoE student
is simultaneously "too large" (total) and "much smaller" (active) than a dense
teacher. The only way to know is to measure. That is why the pilot exists.
