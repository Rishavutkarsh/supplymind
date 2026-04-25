# Black-Box Agent Evaluation Prompt

Use this prompt when evaluating an external agent against SupplyMind without giving it source-code access.

## Goal

Maximize cumulative reward in the live environment by interacting only through the HTTP API.

## Core Principle

Treat SupplyMind as a black-box environment.

You may use any reasoning or planning tools available to you, including calculations, helper code, temporary scripts, or policy notes, but you must not inspect the environment source code, repository files, or hidden implementation details. The HTTP API is the only allowed interface to the environment itself.

## Allowed Endpoints

- `GET /health`
- `POST /reset`
- `GET /state`
- `POST /step`

## Recommended Evaluation Flow

1. Call `GET /health` to confirm the service is live.
2. Start a fresh episode with `POST /reset`.
3. Play the episode entirely through repeated `POST /step` calls until `done = true`.
4. Use `GET /state` only when needed for recovery, inspection, or consistency checks.
5. Base all decisions only on API observations and returned feedback.

## Agent Freedom

The agent is allowed to:
- compute distances, route costs, or heuristics externally
- write temporary helper scripts or planning code
- keep notes or policy summaries across episodes
- retry on new seeds and compare strategies

The agent is not allowed to:
- inspect local repository files or source code
- rely on hidden future schedules or undisclosed reward logic
- modify the environment implementation

## Suggested Curriculum

If you are evaluating learning or strategy improvement across multiple runs:
- start with `cooperative_restock`
- move to `scarcity_negotiation`
- finish on `crisis_coalition`

This keeps the progression aligned with the environment's intended easy -> medium -> hard ladder.

## Final Report

At the end of each evaluation run, report:
- final cumulative reward
- the policy or strategy you followed
- key transfer and coalition decisions
- what the API feedback taught you
- what felt confusing, too easy, too derived, or gameable

