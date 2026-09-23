# LLaVA-NeXT-Video StateRev-VL Replication Handoff

Date: 2026-09-18

## Status

The current cross-model replication scope is complete. The model run, frozen
probe, recursive validation analysis, shard merge, and leakage audits finished.

No circuit localization, attention/head sweep, intervention, rescue, or
patching was run.

## Output Directory

`outputs/vetbench/llava_next_video_7b_replication_v1/`

Main artifacts:

- `behavior.csv`: 250 prefixes, 50 trajectories x t=1..5.
- `hidden_states.npz`: 250 tensors with shape `(33, 4096)`.
- `hidden_event_probe_validation.csv`: validation event accuracy by layer.
- `frozen_hidden_event_decoder.joblib/json`: discovery-fitted decoder.
- `recursive_validation_prefix.csv`: validation recursive predictions.
- `recursive_metrics.csv`: state accuracy by method and subset.
- `recursive_summary.json`: recursive summary and paired comparison.
- `merge_summary.json`: merge and leakage audit.
- `shards/{discovery,validation}/`: four shards per split with completion markers.

## Data Protocol

The exact existing split was reused:

`outputs/vetbench/content_disjoint_split_v1/discovery_validation_split.json`

- Discovery: 30 trajectories, 150 prefixes.
- Validation: 20 trajectories, 100 prefixes.
- Trajectory overlap: 0.
- Cross-split exact sampled pixel fingerprint overlap: 0.
- Cross-split exact hidden tensor overlap: 0.

The original prefix window definition was retained: `[0, frame_end)` for each
t. LLaVA input videos use 16 uniformly spaced frame indices within that window.
Actual frame indices and sampled-frame fingerprints are saved per row.

## Model and Hidden Schema

- Model: `llava-hf/LLaVA-NeXT-Video-7B-hf`
- Transformers: `5.15.1`
- Processor: `LlavaNextVideoProcessor`
- Model class: `LlavaNextVideoForConditionalGeneration`
- Hidden position: final input/prompt token before assistant generation.
- Stored layers: embedding plus 32 transformer layers, total 33 tensors.
- Hidden dimension: 4096.
- Hidden extraction source: state-question forward only.
- Event-question hidden states were not used for probe fitting.

## Probe Protocol

The event probe uses `StandardScaler + L2 LogisticRegression`.

- Fit trajectories: discovery only.
- Validation trajectories: never used for fitting or layer selection.
- Layer selection: 3-fold GroupKFold by trajectory inside discovery.
- Candidate C values: `0.1, 1.0, 10.0`.
- Selected layer: `L1`.
- Selected C: `0.1`.

The selected L1 probe has held-out validation event accuracy `1.000`.

## Validation Results

| Method | Overall | t>=2 | t>=3 | t1 | t2 | t3 | t4 | t5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Native state | 0.280 | 0.300 | 0.317 | 0.200 | 0.250 | 0.100 | 0.400 | 0.450 |
| Explicit-event hard recursion | 0.390 | 0.400 | 0.367 | 0.350 | 0.500 | 0.300 | 0.450 | 0.350 |
| Explicit-event probabilistic recursion | 0.340 | 0.363 | 0.317 | 0.250 | 0.500 | 0.250 | 0.450 | 0.250 |
| Hidden-event hard recursion | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Hidden-event probabilistic recursion | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Oracle-event hard recursion | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

The stored paired difference for hidden hard recursion minus native state at
t>=2 is `+0.700`.

## Important Audit Note

The hidden-event result is unexpectedly strong: selected L1 validation event
accuracy and hidden-event recursive state accuracy are both 100%. Treat this as
an audit target before making a scientific claim.

The merge audit confirms no exact hidden tensor crosses discovery and
validation. Same-split duplicate visual/hidden inputs can exist inside
discovery because the frozen protocol only requires cross-split disjointness.
GPT-5.6 Sol should inspect:

1. Whether LLaVA video features enter the exact state-question forward used for hidden extraction.
2. Whether L1 encodes a temporal/frame-count or prompt artifact correlated with event labels.
3. Whether candidate scoring and generation use identical multimodal inputs and token positions.
4. Whether validation event predictions come only from the frozen decoder.
5. Whether trajectory-level permutation/null analysis changes the apparent 1.0 result.

This is an engineering-complete replication artifact, but not yet a final
mechanistic conclusion without that audit.

## Reproduction Commands

```bash
python scripts/llava_next_video_probe.py
python scripts/llava_next_video_recursive.py
```

Full one-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0 NUM_SHARDS=1 \
bash scripts/run_llava_next_video_replication.sh
```

Four-GPU run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_SHARDS=4 \
bash scripts/run_llava_next_video_replication.sh
```

## Source Files

- `scripts/llava_next_video_replication.py`
- `scripts/llava_next_video_probe.py`
- `scripts/llava_next_video_recursive.py`
- `scripts/run_llava_next_video_replication.sh`

