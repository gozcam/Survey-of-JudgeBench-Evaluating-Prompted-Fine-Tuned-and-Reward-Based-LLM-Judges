# Skywork Critic Local Setup (Ubuntu + GPU)

Notes on running `Skywork/Skywork-Critic-Llama-3.1-8B` locally via vLLM for JudgeBench evaluation.

---

## Environment setup (Ubuntu machine)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Export HuggingFace token so Docker can pull the model weights:

```bash
export HF_TOKEN=your_token_here
```

---

## Verify GPU + Docker access

```bash
docker run --rm --runtime nvidia --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

Should print GPU info. If it errors, the nvidia container toolkit is not configured.

---

## Start the vLLM server

```bash
docker run --runtime nvidia --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 8000:8000 \
  --ipc=host \
  vllm/vllm-openai:latest \
  --model Skywork/Skywork-Critic-Llama-3.1-8B \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

max-model-len and gpu-memory-utilization is what I used w/ 3090ti; adjust depending on hardware capabilities.

The first run downloads the model weights (~16 GB). Subsequent runs reuse the cache at `~/.cache/huggingface`.

Wait for the line:
```
Application startup complete.
```
before running any scripts against it.

---

## Connect from Windows (if Ubuntu is a remote machine)

Forward port 8000 over SSH so the runner scripts can reach it at `localhost:8000`:

```bash
ssh -L 8000:localhost:8000 user@ubuntu-machine
```

Keep this tunnel open while running experiments. No changes to the scripts are needed — they already point to `http://localhost:8000/v1`.

---

## Run the experiments (Windows side)

Activate the Windows venv and run the pilot first to confirm the connection works end-to-end:

```bash
.venv\Scripts\activate
python scripts/runskyworkcritic_pilot.py
```

If the pilot looks correct, run the full dataset:

```bash
python scripts/runskyworkcritic_full.py
```

Output files land in `outputs/` with names like:
```
dataset=judgebench-pilot10,...,judge_name=skywork_critic,judge_model=Skywork_Skywork-Critic-Llama-3.1-8B.jsonl
```

---

## Notes

- The vLLM server uses the OpenAI-compatible API, so no API key is needed — `LocalAPI` in `models.py` sends `api_key="EMPTY"` automatically.
- `--concurrency_limit` is set to 1 in the runner scripts. The vLLM server handles batching internally, so this is safe but can be raised if throughput is too slow.
- If the server goes down mid-run, JudgeBench resumes from where it left off on the next run (it skips already-judged pair IDs).
- Skywork Critic is a fine-tuned judge (not prompted), so it uses `--judge_name skywork_critic` rather than `arena_hard` like the GPT/Gemini scripts.
