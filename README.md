# HPC Task Scheduler

Funny story, this guy, Alex from Boeing (*who apparently has been a full-time stay-at home-dad for the past three years*) interviewed me, and said "I looked at your Github, and it looked like a lot of video games and a lot of music technology. I see a clear pattern here." And so NOW we have a cluster placement simulator that relates to HPC, ML - Reinforcement Learning, and Backend Development. Take that, Alex.

```text
hpc_task_scheduler/
├── include/cluster_env.hpp       # Event-driven C++20 state engine API
├── src/
│   ├── cluster_env.cpp           # Contiguous state, queue, rewards, simulation
│   ├── bindings.cpp              # pybind11 NumPy bridge
│   ├── train_offline_rl.py       # Discrete twin-critic CQL and ONNX export
│   └── main_inference.cpp        # ONNX Runtime benchmark/inference executable
├── hpc_scheduler/
│   ├── __init__.py
│   └── env.py                    # Farama Gymnasium wrapper
├── scripts/
│   ├── data_collector.py         # CSV/Parquet heuristic trajectory collector
│   └── evaluate_policy.py        # Held-out ONNX/baseline replay evaluator
├── tests/
│   ├── test_cluster_env.cpp
│   ├── test_data_collector.py
│   ├── test_evaluator.py
│   ├── test_python_env.py
│   └── test_training_pipeline.py
├── CMakeLists.txt
├── CMakePresets.json
├── pyproject.toml
└── requirements.txt
```

A production-oriented, event-driven cluster placement simulator and offline RL
pipeline. The C++ engine models heterogeneous CPU/RAM nodes, pending jobs, job
arrivals, completions, waiting time, and SLA violations. Its fixed-width state is
stored in one contiguous `std::vector<float>` for cache-efficient native use.

## State, actions, and reward

For `N` nodes and a queue of `K` jobs, the state width is `5N + 6K + 4`:

- Node: CPU utilization, RAM utilization, normalized CPU/RAM capacity, active jobs.
- Queue slot: CPU/RAM request, duration, wait, SLA slack, presence mask.
- Global: normalized time, arrival progress, completion progress, mean utilization.

Discrete actions `0..K*N-1` decode as `queue_slot * N + node`; action `K*N` is
WAIT. Every observation includes an action mask in the Gym `info` dictionary.
The reward implemented by the engine is:

```text
R = -alpha * mean_normalized_queue_wait
    + beta * mean_cluster_utilization
    - gamma * active_SLA_violations
    - invalid_action_penalty
```

## Python installation

Python 3.10–3.13 is recommended (use a PyTorch-supported Python release). A normal
editable install builds only the native simulator extension; ONNX Runtime's C++ SDK
is not needed for data collection or training.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pytest
```

If CMake cannot locate pybind11 in a manual native build, either pass its CMake
directory (`python -m pybind11 --cmakedir`) through `CMAKE_PREFIX_PATH`, or configure
with `-DHPC_FETCH_PYBIND11=ON`.

## Test suites

The Python suite exercises trace sanitation, SJF collection, NPZ/HDF5 round trips,
Gymnasium behavior, invalid-action handling, temporal holdout selection, five-policy
replay, report generation, a one-epoch CQL optimization, verified ONNX export,
dynamic batching, and ONNX Runtime inference:

```powershell
.\.venv\Scripts\python.exe -m pytest -vv
```

The native engine test does not require the ONNX Runtime SDK:

```powershell
.\.venv\Scripts\cmake.exe -S . -B build/native-tests `
  -DHPC_BUILD_PYTHON=OFF -DHPC_BUILD_INFERENCE=OFF -DBUILD_TESTING=ON
.\.venv\Scripts\cmake.exe --build build/native-tests --config Release
.\.venv\Scripts\ctest.exe --test-dir build/native-tests -C Release --output-on-failure
```

For a native inference smoke test, generate a small verified policy with
`tests/generate_test_model.py`, configure with `HPC_BUILD_INFERENCE=ON` and
`ONNXRUNTIME_ROOT`, then run `hpc_inference`. CMake copies the matching ONNX Runtime
shared library beside the executable to prevent an older system DLL from being used.

## Collect an offline dataset

Input traces must contain numeric `submission_time`, `cpu_req`, `mem_req`, and
`duration` columns. An SLA column is optional. Column names are configurable.
Jobs that cannot fit any node are reported and excluded.

```bash
python scripts/data_collector.py data/trace.parquet \
  --nodes "32:128,48:192,64:256" \
  --queue-capacity 32 \
  --heuristic sjf \
  --jobs-per-episode 10000 \
  --output data/offline_dataset.npz
```

Use `--heuristic drf` for dominant-resource ordering. Both NPZ and HDF5 output
contain states, actions, rewards, next states, terminal flags, current/next action
masks, episode IDs, episode returns, and JSON schema metadata.

## Train discrete CQL and export ONNX

```bash
python src/train_offline_rl.py data/offline_dataset.npz \
  --output-dir artifacts \
  --epochs 100 \
  --batch-size 512 \
  --cql-alpha 1.0
```

The trainer uses two critics, Polyak target networks, conservative Q penalties,
entropy-regularized policy improvement, and a behavior-cloning stabilizer. It writes
one JSON metrics record per epoch, compares learned/logged heuristic action values,
checkpoints the best validation policy, and exports observation normalization plus
the actor to `artifacts/scheduler_policy.onnx`.

These Q comparisons are offline estimates, not causal proof of improvement. Before
production deployment, evaluate on held-out time ranges and replay traces in the
simulator to check return, SLA violations, utilization, and invalid-action rate.

## Replay a held-out trace

Evaluate the exported policy and FIFO, SJF, dominant-resource-first, and best-fit
baselines on exactly the same chronological episodes:

```bash
python scripts/evaluate_policy.py data/heldout_trace.parquet \
  artifacts/scheduler_policy.onnx \
  --nodes "32:128,48:192,64:256" \
  --queue-capacity 32 \
  --jobs-per-episode 2000 \
  --holdout-fraction 1.0 \
  --reference sjf \
  --output artifacts/evaluation.json
```

Use a trace or time range that was not used by the collector that produced the
training dataset. When a larger chronological trace is supplied, the evaluator uses
the most recent 20% by default; `--holdout-fraction 1.0` evaluates the entire input.
Model state/action dimensions are checked against the requested node and queue
topology before replay starts.

The JSON summary and sibling per-episode CSV report return, mean/p95 wait, slowdown,
makespan, throughput, SLA violation rate, time-averaged CPU/RAM utilization, invalid
actions, and decision latency. The summary also reports ONNX improvements or
regressions relative to the selected baseline. Resource utilization is calculated as
requested resource-time divided by cluster capacity-time over each episode.

## Native ONNX Runtime build and benchmark

Download an ONNX Runtime C/C++ release and set `ONNXRUNTIME_ROOT` to the directory
containing `include/` and `lib/`. Ensure its shared library is on `PATH` (Windows) or
the platform dynamic-library path at runtime.

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
./build/release/hpc_inference artifacts/scheduler_policy.onnx --iterations 10000
```

On multi-config Windows generators the executable may be under `Release/`. The
benchmark pins ONNX Runtime intra/inter-op thread counts to one, enables all graph
optimizations, performs warmup, and reports mean/p50/p95/p99 latency plus whether
p95 met the 200 microsecond target. The target must be validated on deployment
hardware; it cannot be guaranteed by source code alone.

## Deployment contract

The ONNX model accepts both the contiguous state and the `uint8` form of the boolean
mask returned by `ClusterEnv::action_mask()`. It emits already-masked action logits, so serving takes
`argmax` without allowing an impossible placement. Keep the node ordering, queue
capacity, state schema, and action mapping identical to the dataset metadata. Route
model-load or inference failures to a deterministic SJF/DRF fallback rather than
dropping jobs.
