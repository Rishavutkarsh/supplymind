# Hackathon Requirements Reference

This document distills the important implementation and submission requirements captured from the saved hackathon page:

- Source page: `C:\Users\risha\Desktop\Scaler School of Technology.html`
- Page title: `Meta PyTorch OpenEnv Hackathon` dashboard

This file is intended to be a clean reference for validator-facing and submission-facing requirements. It is separate from `PROJECT_SPEC.md`, which describes the project design.

## Additional Organizer Clarifications

From Discord guidance shared during the hackathon:
- keep both the GitHub repository and Hugging Face Space public for evaluation
- implement the standard OpenEnv endpoints such as `/reset`, `/step`, and `/state`
- use environment variables for runtime configuration like `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`
- local `.env` usage is acceptable for development
- do not commit private API keys
- do not assume tokens will be available during evaluation
- the submission should be self-contained and reproducible
- the submission should not break if tokens are missing
- using an external LLM/API in the baseline is not mandatory if the inference path is valid and reproducible

## Timeline

- Registration: `14th March - 3rd April`
- Prepare: `Now - 25th March`
- Round 1: `25th March - 8th April`
- Results: `10th April`
- Finale: `25th-26th April`
- Submission window opens: `28th March`
- Round 1 deadline: `8 Apr 11:59 PM`

## Round 1 Deliverable

Build and submit a real-world `OpenEnv` environment.

The page emphasizes:
- not a toy or purely gimmicky environment
- clear, realistic, testable tasks
- automated grading
- meaningful reward logic
- OpenEnv packaging for automated evaluation

Examples on the page frame this as building a practical agent environment rather than a game-only project.

## Required Environment Contract

The implementation must provide:
- `reset() -> initial observation`
- `state() -> current state`
- `step(action) -> observation, reward, done, info`

The page also calls out typed models for:
- `Observation`
- `Action`
- `Reward`

These should be implemented with Pydantic for submission readiness.

## Required Packaging / Files

The page requires or strongly implies the submission must include:
- root-level `inference.py`
- `openenv.yaml`
- working `Dockerfile`
- `README`

The `README` should cover:
- environment description
- action space
- observation space
- task descriptions and difficulty
- setup and usage instructions
- baseline scores

## Task / Grading Requirements

- define at least `3` tasks
- tasks should have increasing difficulty
- each task must have a grader
- each grader must output a score in the range `0.0` to `1.0`
- reward logic should be meaningful, not just binary end-state scoring

The page's evaluation framing emphasizes:
- runtime correctness
- interface compliance
- task design
- grading logic

## Validation / Disqualification Risk

The saved page and extracted requirements indicate the following checks are validator-critical:

- Hugging Face Space URL is checked automatically
- the Space must return `200`
- the Space must respond to `reset()`
- `openenv.yaml` is validated
- OpenEnv interface compliance is checked
- Docker build is checked
- submitted `inference.py` is executed
- graders are run across tasks
- each grader score must be within `0.0-1.0`

Important implication:
- all validator-facing checks should be treated as mandatory
- packaging correctness is as important as the simulator itself

## Runtime / Infra Constraints

The extracted requirements indicate:
- all LLM calls should use `OpenAI Client`
- required environment variables:
  - `API_BASE_URL`
  - `MODEL_NAME`
  - `HF_TOKEN`
- target runtime for inference: under `20 minutes`
- target machine: `2 vCPU`, `8 GB RAM`

Practical interpretation after organizer clarification:
- support env-var-driven external model usage cleanly
- do not require tokens for the submission to execute
- provide a reliable self-contained path for evaluation
- use the OpenAI client for all LLM calls
- for this repo, prefer `HF_TOKEN` over `OPENAI_API_KEY` when both are present
- the active submission backend is `v3`, not the older `v1` delivery path

Observed integration note during implementation:
- the Hugging Face OpenAI-compatible path can be configured correctly with env vars
- token auth and endpoint wiring can work, but provider usage may still fail if Hugging Face inference credits are depleted
- this reinforces the need for a reliable self-contained submission path

## HF Space Expectations

The submission should be deployable to Hugging Face Spaces and expose a validator-friendly interface.

For planning purposes, we are treating the following as required:
- service starts successfully
- validator can reach it over HTTP
- `reset()` is callable successfully
- the API surface stays aligned with the same typed environment schema used locally

## Practical Build Priorities

These requirements imply the implementation should prioritize:

1. validator-safe typed interfaces
2. `openenv.yaml` and packaging correctness
3. deterministic tasks and graders
4. reliable root `inference.py`
5. thin HF Space API wrapper
6. polished UX only after validator readiness

## Notes

- This file is intentionally a requirements reference, not the product design spec.
- The delivery-dispatch environment design lives in `PROJECT_SPEC.md`.
- If we discover exact validator wire-format examples later, this document should be updated to reflect them.

