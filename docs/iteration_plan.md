# SupplyMind Iteration Plan

This document captures the next disciplined improvement loop for SupplyMind.

## Iteration Principles

1. Change one thing at a time.
2. Evaluate on fixed train seeds and held-out eval seeds.
3. Judge progress by both score and behavior.
4. Keep the environment black-box from the agent's perspective.

## Seed Discipline

Use small repeated seed sets instead of one-off runs.

Suggested split:
- train seeds: `1, 2, 3`
- eval seeds: `101, 102, 103`

Use the same seeds before and after each change.

## Curriculum

Run experiments in this order:
- `cooperative_restock` for API and policy sanity
- `scarcity_negotiation` for strategy formation
- `crisis_coalition` for dynamic robustness

The goal is not to learn only on the hardest task.

## Iteration 1: Better End-of-Run Learning Signal

Hypothesis:
- Agents need a clearer terminal summary to improve across repeated episodes.

Success criteria:
- agents can explain what caused score loss
- policy revisions become more targeted

Candidate additions:
- reward lost to stockouts
- reward lost to transfer friction
- invalid inventory proposals
- missed coalition opportunities

## Iteration 2: Reduce Passive Stock Hoarding

Hypothesis:
- repeated no-transfer decisions can be too attractive when transfer friction is high.

Success criteria:
- agents stop gaining from passive stock hoarding
- strategic reserve holding remains viable

Candidate changes:
- track avoidable stockouts when surplus exists elsewhere
- penalize only when a feasible coalition transfer was available
- keep reserve behavior legal when future demand risk is plausible

## Iteration 3: Make Medium/Hard More Distinct Strategically

Hypothesis:
- `scarcity_negotiation` and `crisis_coalition` should differ not only in pressure, but in the kind of planning they demand.

Success criteria:
- medium rewards truthful scarcity management
- hard rewards coalition formation under demand shocks and transfer friction

Candidate changes:
- keep `scarcity_negotiation` steady but inventory-constrained
- make `crisis_coalition` reward anticipation of late premium demand more strongly

## Evaluation Protocol

For each iteration:

1. Run baseline and target on train/eval seeds.
2. Run one black-box agent with the prompt in `docs/agent_eval_prompt.md`.
3. Compare:
- cumulative reward
- stockout rate
- transfer friction cost
- invalid proposals
- qualitative strategy

Keep the change only if both behavior and metrics improve.

