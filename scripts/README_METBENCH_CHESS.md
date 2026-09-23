# MET-Bench Chess: single-piece tracking

`metbench_chess_single_piece.py` is the first StateRev-VL implementation for
the chess state task. It discovers canonical `full-*.parquet` shards, checks
the nested schema, and writes
`outputs/metbench_chess/single_piece_tracking_v1/manifest.csv`.

The target is the white knight initially on `g1`. A prefix is one ordinary UCI
move (`t` starts at 1); `src` and `dst` are the current move, and
`current_state` is a square or `captured`. A game is truncated immediately
before the first castling, en-passant, promotion, malformed, or illegal move.
The manifest stores a `parquet://...#row=...&action=...` visual reference and
an exact PNG SHA-256 fingerprint, so the 12 GB visual dataset is not copied.

## Smoke and audits

```bash
python scripts/metbench_chess_single_piece.py build-manifest
python scripts/metbench_chess_single_piece.py audit
python scripts/metbench_chess_single_piece.py unit
python scripts/metbench_chess_single_piece.py special-unit
python scripts/metbench_chess_single_piece.py shard-unit
python scripts/metbench_chess_single_piece.py model-smoke \
  --model qwen --model-dir models/Qwen3-VL-8B-Instruct --limit 2
python scripts/metbench_chess_single_piece.py model-smoke \
  --model llava --model-dir models/LLaVA-NeXT-Video-7B-hf --limit 2
python scripts/metbench_chess_single_piece.py native-smoke \
  --model qwen --model-dir models/Qwen3-VL-8B-Instruct --limit 1
python scripts/metbench_chess_single_piece.py native-smoke \
  --model llava --model-dir models/LLaVA-NeXT-Video-7B-hf --limit 1
```

The model smoke asks only the state question and extracts the final input
hidden state. It does not use move-question hidden states. Missing model/GPU
dependencies are reported as `SMOKE_SKIPPED`, never as a false pass.
`native-smoke` additionally generates one native state answer and one explicit
move answer; it writes both texts with separate `question_type` fields.

## CPU shard/merge check

```bash
python scripts/metbench_chess_single_piece.py manifest-shard \
  --manifest outputs/metbench_chess/single_piece_tracking_v1/manifest.csv \
  --shard-index 0 --num-shards 2
python scripts/metbench_chess_single_piece.py manifest-shard \
  --manifest outputs/metbench_chess/single_piece_tracking_v1/manifest.csv \
  --shard-index 1 --num-shards 2
python scripts/metbench_chess_single_piece.py manifest-merge --num-shards 2
```

Each shard has an independent `.complete.json` marker and SHA-256; merge
refuses missing, corrupted, duplicated, or incomplete game coverage.

## Hidden square probes

After a state-question extraction has produced an `.npz` cache with keys
`<game_id>_t<t>` (and a sidecar identifying `question_type=state`), fit the
discovery-only source/destination probes with:

```bash
python scripts/metbench_chess_single_piece.py fit-probe \
  --hidden-cache outputs/metbench_chess/single_piece_tracking_v1/qwen_hidden.npz \
  --candidate-layers 0,12,24,36 --candidate-c 0.1,1,10
```

The probe is `StandardScaler + LogisticRegression`, with separate 64-way
source and destination classifiers and a frozen discovery-only layer/C choice.
`ChessStateUpdater` has 8,353 trainable parameters (64 squares + captured);
the VLM and probes are frozen. `recursive_state_tracking(..., mode="oracle")`
is the handwritten-transition upper bound used only for diagnostics.
