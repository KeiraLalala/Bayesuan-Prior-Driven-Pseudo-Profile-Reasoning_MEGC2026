# ME-LVQA / ME-VQA Train-Only Prior-Control Filler

## Project Overview
This project fills target ME-LVQA and ME-VQA question JSONL files using a train-only prior-control pipeline:
1. build reference profiles from labeled/answered training data,
2. estimate empirical priors,
3. parse target requirements,
4. construct one pseudo-profile per target video,
5. decode answers from that shared profile.

Qwen is used as a target-side language interface (question parsing and optional VQA analysis rendering).

## Project Structure
```text
project_root/
|-- run_me_lvqa_fill.py
|-- run_me_vqa_fill.py
|-- model/
|   `-- qwen_bridge.py
|-- util/
|   |-- io_utils.py
|   |-- generate_lvqa_constants_json.py
|   `-- generate_vqa_constants_json.py
|-- me_lvqa_util/
|   |-- config_constants.json
|   |-- lvqa_reference.py
|   |-- lvqa_priors.py
|   |-- lvqa_requirements.py
|   `-- lvqa_inference.py
|-- me_vqa_util/
|   |-- vqa_config_constants.json
|   |-- vqa_reference.py
|   |-- vqa_priors.py
|   |-- vqa_requirements.py
|   `-- vqa_inference.py
`-- Q&A_jsonl/
```

## Main Pipeline
- Stage 1 (`*_reference.py`): reference profile reconstruction from training/reference annotations.
- Stage 2 (`*_priors.py`): empirical prior estimation from reference profiles.
- Stage 3 (`*_requirements.py`): target question parsing and requirement extraction.
- Stage 4 (`*_inference.py`): pseudo-profile construction + answer decoding + output writing.


## How to Run
- LVQA:
```bash
uv run python run_me_lvqa_fill.py
```

- VQA:
```bash

uv run python run_me_vqa_fill.py --use-qwen
```

## Input Files
- LVQA:
  - Train answered JSONL: `Q&A_jsonl/me_lvqa_samm_casme3.jsonl`
  - Target JSONL: `Q&A_jsonl/me_lvqa_casme3_test_to_answer.jsonl`, `Q&A_jsonl/me_lvqa_samm_test_to_answer.jsonl`
- VQA:
  - Reference labeled JSONL: `Q&A_jsonl/me_vqa_samm_casme2_smic_v2.jsonl`
  - Target JSONL: `Q&A_jsonl/me_vqa_casme3_v2_test_to_answer.jsonl`, `Q&A_jsonl/me_vqa_samm_v2_test_to_answer.jsonl`

## Output
- LVQA script writes:
  - `me_lvqa_casme3_test_pred.jsonl`
  - `me_lvqa_samm_test_pred.jsonl`
- VQA script writes:
  - `me_vqa_casme3_v2_test_pred.jsonl`
  - `me_vqa_samm_v2_test_pred.jsonl`

All outputs are written under `--outdir`.

## Notes
- Determinism: profile sampling uses seeded stable RNG (`--seed`).
- Config constants:
  - LVQA: `me_lvqa_util/config_constants.json` (generate/update with `uv run python util/generate_lvqa_constants_json.py`)
  - VQA: `me_vqa_util/vqa_config_constants.json` (generate/update with `uv run python util/generate_vqa_constants_json.py`)
- Environment caveat: full LVQA run requires Qwen backend availability (`transformers` + model access).
