# Survey of JudgeBench

This repository contains the standalone scripts and small code changes used to run my survey experiments on **JudgeBench**, focused on comparing different kinds of LLM judges.

The main goal is to evaluate how well different judge models act as evaluators on JudgeBench, especially across:
- **Knowledge** (`mmlu-pro`)
- **Reasoning** (`livebench-reasoning`)
- **Math** (`livebench-math`)
- **Coding** (`livecodebench`)

The project is structured so it is **not just a copy of the original JudgeBench repo**. Instead, it uses standalone runner scripts and local modifications that make the experiments easier to reproduce and analyze for the report.

---

## Project focus

This survey studies three judge families:

- **Prompted judges**  
  General LLMs used with a judging prompt, such as:
  - GPT-4o-mini
  - Gemini Flash Lite
  - other prompted baselines if added later

- **Fine-tuned judges**  
  Models explicitly trained for judging / critique

- **Reward models**  
  Models that score both responses and select the higher-scored one

The current repository already supports the prompted-judge pipeline and can be extended with additional models.

---

## Repository structure

A typical layout looks like this:

```text
repo/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ dataset=judgebench-pilot10,response_model=gpt-4o-2024-05-13.jsonl
│  └─ ...
├─ outputs/
│  ├─ dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name=arena_hard,judge_model=gpt-4o-mini.jsonl
│  └─ ...
├─ scripts/
│  ├─ rungpt4omini_pilot.py
│  ├─ rungpt4omini_full.py
│  ├─ rungeminiflashlite_pilot.py
│  └─ rungeminiflashlite_full.py
└─ third_party/
   └─ judgebench/
      ├─ run_judge.py
      ├─ utils/
      │  ├─ file_operations.py
      │  ├─ judges.py
      │  ├─ metrics.py
      │  ├─ models.py
      │  └─ prompts.py
      └─ data/
         └─ dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl
```

---

## What was changed from the upstream JudgeBench setup

This repo includes practical changes for reproducibility:

- standalone pilot and full-run scripts for each model
- pilot subsets saved under root `data/`
- judged outputs written to root `outputs/`
- UTF-8 output writing fix for Windows compatibility
- metrics guard for empty-category pilot subsets
- environment-specific setup notes for API-based runs

---

## Setup

### 1. Create and activate a virtual environment

### Windows Command Prompt
```cmd
python -m venv .venv
.venv\Scripts\activate
```

### Windows PowerShell
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### macOS / Linux / WSL
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If needed for Gemini support:

```bash
pip install google-generativeai
```

If needed for OpenAI/httpx compatibility, pin:

```bash
pip install httpx==0.27.2
```

---

## API keys

Export the required key before running a model.

### OpenAI (GPT-4o-mini)

#### Windows Command Prompt
```cmd
set OPENAI_API_KEY=your_key_here
```

#### Windows PowerShell
```powershell
$env:OPENAI_API_KEY="your_key_here"
```

#### macOS / Linux / WSL
```bash
export OPENAI_API_KEY="your_key_here"
```

### Gemini Flash Lite

#### Windows Command Prompt
```cmd
set GEMINI_API_KEY=your_key_here
```

#### Windows PowerShell
```powershell
$env:GEMINI_API_KEY="your_key_here"
```

#### macOS / Linux / WSL
```bash
export GEMINI_API_KEY="your_key_here"
```

---

## Required dataset

The full JudgeBench dataset used in these runs is expected at:

```text
third_party/judgebench/data/dataset=judgebench,response_model=gpt-4o-2024-05-13.jsonl
```

Pilot scripts generate a smaller 10-pair subset and save it under:

```text
data/dataset=judgebench-pilot10,response_model=gpt-4o-2024-05-13.jsonl
```

---

## Running experiments

## GPT-4o-mini pilot

```bash
python scripts/rungpt4omini_pilot.py
```

This:
1. creates or reuses a 10-pair pilot subset
2. runs JudgeBench using GPT-4o-mini as the judge
3. writes judged results into root `outputs/`

## GPT-4o-mini full run

```bash
python scripts/rungpt4omini_full.py
```

This runs the full JudgeBench GPT-4o response-pair dataset.

## Gemini Flash Lite pilot

```bash
python scripts/rungeminiflashlite_pilot.py
```

## Gemini Flash Lite full run

```bash
python scripts/rungeminiflashlite_full.py
```

If your scripts live in the repo root instead of `scripts/`, use:

```bash
python rungpt4omini_pilot.py
python rungpt4omini_full.py
python rungeminiflashlite_pilot.py
python rungeminiflashlite_full.py
```

---

## Output files

Judged results are written to the root `outputs/` folder.

Examples:

```text
outputs/dataset=judgebench-pilot10,response_model=gpt-4o-2024-05-13,judge_name=arena_hard,judge_model=gpt-4o-mini.jsonl
outputs/dataset=judgebench,response_model=gpt-4o-2024-05-13,judge_name=arena_hard,judge_model=gemini-2.5-flash-lite.jsonl
```

These JSONL files contain the original pair fields plus:
- `judge_name`
- `judgments`

They are the source of truth for later analysis.

---

## Seeing the category breakdown

At the end of a run, `run_judge.py` prints:
- `mmlu-pro`
- `livebench-reasoning`
- `livebench-math`
- `livecodebench`
- `Overall`

These correspond to:
- Knowledge
- Reasoning
- Math
- Coding
- Overall accuracy

---

## Inspecting failures

To inspect specific failures after a run, read the saved JSONL in `outputs/` and compare:
- gold label: `label`
- model decision(s): `judgments`

Recommended approach:
- write a small inspection script that filters incorrect pairs
- group failures by `source`
- save representative mistakes for the report discussion section

Useful fields per row:
- `pair_id`
- `source`
- `question`
- `response_A`
- `response_B`
- `label`
- `judgments`

---

## Reproducibility notes

### Important local fixes
These were important during setup and may be needed again:

- **Windows UTF-8 fix:** output files should be written with `encoding="utf-8"`
- **OpenAI/httpx compatibility:** `httpx==0.27.2` may be required
- **Pilot subset location:** saved under root `data/`
- **Run outputs:** written to root `outputs/`

### Rerun behavior
If an output file already exists, `run_judge.py` skips already judged `pair_id`s and resumes from the remaining ones.

This is useful for interrupted runs, but if you want a completely clean rerun, delete the old output file first.

---

## Resources used

This project mainly uses:
- Python virtual environment
- OpenAI API for GPT-4o-mini experiments
- Gemini API for Gemini Flash Lite experiments

These runs are API-based, so local GPU usage is not required for the current prompted-judge experiments.

---

## Report mapping

This codebase supports the required report sections as follows:

- **Introduction:** JudgeBench and LLM-as-a-judge framing
- **Experimental Setup:** scripts, datasets, judge models, environment variables, API setup
- **Results:** output JSONL files and category metrics
- **Discussion:** saved failure cases and cross-model comparisons
- **Resources:** API use and runtime notes
- **Code:** this repository and these reproducible run commands

---

## Common troubleshooting

### `OPENAI_API_KEY is not set`
Export the OpenAI key in the same terminal session before running the GPT-4o-mini script.

### `GEMINI_API_KEY is not set`
Export the Gemini key in the same terminal session before running the Gemini script.

### `AsyncClient.__init__() got an unexpected keyword argument 'proxies'`
Pin:

```bash
pip install httpx==0.27.2
```

### Unicode / `cp1252` write errors on Windows
Make sure output files are opened with:

```python
open(output_file, "a", encoding="utf-8")
```

### Pilot metrics crash due to empty category
If a very small pilot subset misses a category entirely, the metrics function should guard against divide-by-zero.

---

## Future extensions

This repository is meant to grow as more judge models are added. The plan is to keep one reference log per model run, then combine those logs into the final survey report.

Potential additions:
- more prompted judges
- fine-tuned critics
- reward models
- automated analysis scripts for failure extraction and table generation
