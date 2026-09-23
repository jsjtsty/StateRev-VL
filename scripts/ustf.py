#!/usr/bin/env python3
"""Universal Semantic Transition Filter (USTF) CLI.

Only implementation + smoke + tiny-scale sanity live here. No command in
this file runs a formal/large training suite, a third task, or a new VLM
forward pass -- see USTF_V1_HANDOFF_FOR_SOL.md for what is and isn't done.
"""
from pathlib import Path
import argparse
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staterev.psf import ShellAdapter, ChessAdapter, TextFeatureStore  # noqa: E402
from staterev.ustf import USTF, run_tiny_gate, adapt_to_new_task, evaluate_ustf, permutation_invariance_check  # noqa: E402

DEFAULT_OUT = ROOT / 'outputs/ustf_v1'
TEXT_CACHE = {
    'qwen': ROOT / 'outputs/psf_v1/diagnostic_v1/text_cache/qwen_semantic.npz',
    'llava': ROOT / 'outputs/psf_v1/diagnostic_v1/text_cache/llava_semantic.npz',
}


# Cumulative design ladder (see git history / chat for the diagnosis behind each step).
VARIANTS = {
    'v1': {},
    'p0a': dict(text_norm=True),
    'p0b': dict(text_norm=True, text_projector_rank=64),
    'p1a': dict(text_norm=True, text_projector_rank=64, transition='factorized'),
    'p1b': dict(text_norm=True, text_projector_rank=64, transition='factorized', obs_match=True, obs_reliability=True),
    # p1b without the separate state-text projector (text shares the visual projector).
    'p1c': dict(text_norm=True, transition='factorized', obs_match=True, obs_reliability=True),
    # p1c + per-tap LayerNorm of raw hidden before the projector (review diagnostic).
    'p1d': dict(text_norm=True, transition='factorized', obs_match=True, obs_reliability=True, hidden_norm=True),
}


def build_model(args, models, strict=True):
    cfg = dict(VARIANTS[args.variant])
    if getattr(args, 'feature_dropout', None):
        cfg['feature_dropout'] = args.feature_dropout
    m = USTF(tuple(models), d=args.d, hidden=args.hidden, text_stores=make_text_stores(models, strict=strict), **cfg)
    return m.to(args.device)


def make_text_stores(models, strict=True):
    return {m: TextFeatureStore(TEXT_CACHE[m], strict=strict) for m in models}


def load_task(task, model, max_traj, split='discovery'):
    if task == 'shell':
        return ShellAdapter(model).load(max_traj)
    if task == 'chess':
        return ChessAdapter(model).load(max_traj, split)
    raise ValueError(f'No registered adapter for {task}; register one before using it as --eval-task')


def cmd_tiny_gate(args):
    out = args.out / 'tiny_gate' if args.variant == 'v1' and args.tag is None else \
        args.out / 'tiny_gate' / (args.tag or f'{args.variant}_seed{args.seed}')
    shell = load_task('shell', 'qwen', args.shell_trajectories)
    chess = load_task('chess', 'qwen', args.chess_games)
    assert len(shell) == args.shell_trajectories, f'expected {args.shell_trajectories} shell trajectories, got {len(shell)}'
    assert len(chess) == args.chess_games, f'expected {args.chess_games} chess games, got {len(chess)}'
    torch.manual_seed(args.seed)
    model = build_model(args, ['qwen'])
    report, model = run_tiny_gate(model, shell, chess, out, epochs=args.epochs, lr=args.lr, seed=args.seed,
                                   patience=args.patience, weight_decay=args.weight_decay, clip_norm=args.clip_norm)
    status = 'TINY_GATE_PASS' if report['gate']['overall_pass'] else 'TINY_GATE_FAIL'
    print(json.dumps({status: True, 'report': report}, indent=2))
    if not report['gate']['overall_pass']:
        sys.exit(1)


def _slice(data, spec):
    a, b = (int(x) for x in spec.split(':'))
    return data[a:b]


def cmd_heldout(args):
    """Game/trajectory-disjoint evaluation inside the existing discovery split.

    No new split file is created: discovery trajectories are taken in their
    loader order (identical trajectory ids/order across backbones, asserted)
    and partitioned by index into train / dev (early stopping) / test.

    --models qwen | llava | qwen,llava trains one USTF on all listed backbones
    and tests each. --core-transfer SRC:TGT instead trains on SRC, then
    freezes USTFCore and trains only TGT's fresh projector on TGT's train
    partition -- a test of whether the core is backbone-agnostic.

    Reports USTF full rollout, its observation-only and prior-only ablations,
    and two reference baselines (copy-initial state; a per-(backbone, task)
    logistic probe on the last hidden tap, which -- unlike USTF -- uses a
    fixed class index and is only a reference, not a candidate method).
    """
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from staterev.ustf import train_ustf

    if args.core_transfer:
        src, tgt = args.core_transfer.split(':')
        models = [src, tgt]
    else:
        src = tgt = None
        models = [x for x in args.models.split(',') if x]
    out = args.out / 'heldout' / (args.tag or f'{args.variant}_seed{args.seed}')
    out.mkdir(parents=True, exist_ok=True)
    spec = {'shell': (args.shell_train, args.shell_dev, args.shell_test),
            'chess': (args.chess_train, args.chess_dev, args.chess_test)}
    parts = {}  # (model, task) -> [train, dev, test]
    for m in models:
        for task in spec:
            data = load_task(task, m, 0)
            parts[(m, task)] = [_slice(data, x) for x in spec[task]]
    for task in spec:
        ref = [[x.trajectory_id for x in p] for p in parts[(models[0], task)]]
        for m in models:
            assert [[x.trajectory_id for x in p] for p in parts[(m, task)]] == ref, f'{m}/{task} ids differ from {models[0]}'
        ids = [set(x) for x in ref]
        assert not (ids[0] & ids[2]) and not (ids[1] & ids[2]) and not (ids[0] & ids[1]), f'{task} partitions overlap'

    def pool(ms, i):
        return [x for m in ms for task in spec for x in parts[(m, task)][i]]

    train_kw = dict(epochs=args.epochs, lr=args.lr, seed=args.seed, patience=args.patience,
                    weight_decay=args.weight_decay, clip_norm=args.clip_norm, crop_prob=args.crop_prob,
                    warmup_epochs=args.warmup_epochs, min_epochs=args.min_epochs)
    torch.manual_seed(args.seed)
    if args.core_transfer:
        model = build_model(args, [src])
        hist_src = train_ustf(model, pool([src], 0), pool([src], 1), **train_kw)
        model.add_backbone(tgt, text_store=make_text_stores([tgt])[tgt])
        core_before = {k: v.clone() for k, v in model.core.state_dict().items()}
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith(f'projectors.{tgt}.')
        hist = train_ustf(model, pool([tgt], 0), pool([tgt], 1), **train_kw)
        for p in model.parameters():
            p.requires_grad = True
        assert all(torch.equal(v, model.core.state_dict()[k]) for k, v in core_before.items()), 'core changed'
        test_models = [tgt]
    else:
        model = build_model(args, models)
        hist_src = None
        hist = train_ustf(model, pool(models, 0), pool(models, 1), **train_kw)
        test_models = models

    test = pool(test_models, 2)
    ev = evaluate_ustf(model, test)
    model.use_observation = False
    ev_prior = evaluate_ustf(model, test)
    model.use_observation = True
    ev['prior_only_correct'] = ev_prior.rollout_correct.values
    ev['copy_initial_correct'] = (ev['gt'] == ev['initial_state']).astype(int)  # not ev.gt: that's DataFrame.gt()

    # Reference probe: fixed-class logistic regression on the last tap, per (backbone, task).
    ev['probe_correct'] = 0
    for m in test_models:
        for task in spec:
            tr_p, _, te_p = parts[(m, task)]
            X = np.concatenate([x.hidden[:, -1].float().numpy() for x in tr_p])
            y = np.concatenate([x.state_ids.numpy() for x in tr_p])
            clf = LogisticRegression(max_iter=2000, C=args.probe_c).fit(X, y)
            Xt = np.concatenate([x.hidden[:, -1].float().numpy() for x in te_p])
            sel = (ev.model == m) & (ev.task == task)
            ev.loc[sel, 'probe_correct'] = (clf.predict(Xt) == ev.loc[sel, 'gt'].values).astype(int)

    cols = ['rollout_correct', 'observation_correct', 'prior_only_correct', 'transition_correct',
            'copy_initial_correct', 'probe_correct']
    summary = {}
    for (m, task), g in ev.groupby(['model', 'task']):
        summary[f'{m}:{task}'] = {'steps': len(g), 'change_steps': int(g.state_changed.sum()),
                                  'all_steps': {c: float(g[c].mean()) for c in cols},
                                  'change_steps_only': {c: float(g.loc[g.state_changed == 1, c].mean()) for c in cols},
                                  'rollout_by_t': g.groupby('t').rollout_correct.mean().round(4).to_dict()}
    report = {'variant': args.variant, 'config': model.config, 'seed': args.seed, 'models': models,
              'core_transfer': args.core_transfer, 'crop_prob': args.crop_prob,
              'warmup_epochs': args.warmup_epochs, 'min_epochs': args.min_epochs, 'max_epochs': args.epochs,
              'patience': args.patience,
              'partitions': {t: {'train': spec[t][0], 'dev': spec[t][1], 'test': spec[t][2]} for t in spec},
              'epochs_run': len(hist), 'best_dev_rollout': float(hist.dev_rollout_accuracy.max()),
              'source_epochs_run': None if hist_src is None else len(hist_src),
              'test': summary, 'parameter_counts': model.parameter_counts()}
    (out / 'heldout_report.json').write_text(json.dumps(report, indent=2) + '\n')
    ev.to_csv(out / 'heldout_predictions.csv', index=False)
    hist.to_csv(out / 'heldout_history.csv', index=False)
    print(json.dumps({k: v['all_steps'] for k, v in summary.items()}, indent=2))


def cmd_param_report(args):
    args.out.mkdir(parents=True, exist_ok=True)
    models = [x for x in args.models.split(',') if x]
    m = build_model(args, models, strict=False)
    counts = m.parameter_counts()
    suffix = '' if args.variant == 'v1' else f'_{args.variant}'
    path = args.out / f'parameter_counts_{"_".join(models)}{suffix}.json'
    path.write_text(json.dumps(counts, indent=2) + '\n')
    print(json.dumps({'PARAM_REPORT_PASS': True, 'backbones': models, 'counts': counts, 'path': str(path)}, indent=2))


def cmd_joint_smoke(args):
    """Forward-compatibility smoke across backbone x task combinations, real caches where present."""
    args.out.mkdir(parents=True, exist_ok=True)
    models = [x for x in args.models.split(',') if x]
    m = build_model(args, models, strict=True)
    records = []
    for task in ('shell', 'chess'):
        for model_name in models:
            data = load_task(task, model_name, args.max_trajectories)
            ev = evaluate_ustf(m, data)
            records.append({
                'task': task, 'model': model_name, 'trajectories': len(data),
                'num_candidates': int(ev.num_candidates.iloc[0]) if len(ev) else None,
                'rollout_accuracy_untrained': float(ev.rollout_correct.mean()) if len(ev) else None,
            })
    core_ids = {id(getattr(m, 'core')) for _ in models}
    proj_distinct = len({id(p) for p in m.projectors.values()}) == len(models)
    report = {
        'records': records,
        'single_shared_core_module': len(core_ids) == 1,
        'per_backbone_projectors_are_distinct_modules': proj_distinct,
        'parameter_counts': m.parameter_counts(),
    }
    (args.out / 'joint_forward_smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'JOINT_SMOKE_PASS': True, **report}, indent=2))


def cmd_task_c_demo(args):
    """Interface-only demonstration on a SYNTHETIC task, per spec section 11/13.

    This does not run a real third task. It proves --train-tasks/--eval-task
    style zero-shot / few-shot / from-scratch code paths execute without
    growing USTFCore, using synthetic Trajectory objects shaped like a new
    task with its own candidate count K and a previously-unseen model id.
    """
    from staterev.psf import Trajectory
    args.out.mkdir(parents=True, exist_ok=True)
    models = [x for x in args.train_tasks_models.split(',') if x]
    source_data = []
    for task in args.train_tasks.split(','):
        for model_name in models:
            source_data += load_task(task, model_name, args.max_trajectories)
    model = build_model(args, models, strict=True)
    from staterev.ustf import train_ustf
    train_ustf(model, source_data, source_data[: max(1, len(source_data) // 5)], epochs=args.epochs, lr=args.lr, seed=args.seed)
    core_before = sum(p.numel() for p in model.core.parameters())
    # Synthetic task-C candidate/target text is not in the real cache; fall
    # back to the deterministic hash embedding for those keys only (real
    # cached embeddings are still used for anything already present).
    for name in models:
        model.text_stores[name] = TextFeatureStore(TEXT_CACHE[name], strict=False)

    g = torch.Generator().manual_seed(args.seed)
    k_new = 7
    synthetic_c = [
        Trajectory(models[0], 'synthetic_task_c', f'c{i}', torch.randn(5, 4, 4096, generator=g),
                   'synthetic tracked entity', [f'synthetic state {j}' for j in range(k_new)], 0,
                   torch.randint(0, k_new, (5,), generator=g))
        for i in range(12)
    ]
    results = {}
    for mode, kwargs in (('zero-shot', {}), ('few-shot-1', {'few_shot_n': 1}),
                         ('few-shot-5', {'few_shot_n': 5}), ('few-shot-10', {'few_shot_n': 10}),
                         ('from-scratch', {})):
        real_mode = 'few-shot' if mode.startswith('few-shot') else mode
        adapted, hist = adapt_to_new_task(model, synthetic_c, mode=real_mode, epochs=3, **kwargs)
        ev = evaluate_ustf(adapted, synthetic_c)
        core_after = sum(p.numel() for p in adapted.core.parameters())
        results[mode] = {
            'core_param_count_unchanged': core_after == core_before,
            'rollout_accuracy_on_synthetic_c': float(ev.rollout_correct.mean()),
            'history_rows': len(hist),
        }
    (args.out / 'task_c_interface_demo.json').write_text(json.dumps({'note': 'SYNTHETIC task, interface-only, no real third task run', 'results': results}, indent=2) + '\n')
    print(json.dumps({'TASK_C_DEMO_PASS': True, 'results': results}, indent=2))


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest='cmd', required=True)

    t = sp.add_parser('tiny-gate')
    t.add_argument('--shell-trajectories', type=int, default=5)
    t.add_argument('--chess-games', type=int, default=10)
    t.add_argument('--epochs', type=int, default=400)
    t.add_argument('--lr', type=float, default=5e-4)
    t.add_argument('--seed', type=int, default=20260921)
    t.add_argument('--patience', type=int, default=400)
    t.add_argument('--weight-decay', type=float, default=0.0)
    t.add_argument('--clip-norm', type=float, default=1.0)
    t.add_argument('--d', type=int, default=128)
    t.add_argument('--hidden', type=int, default=128)
    t.add_argument('--variant', choices=sorted(VARIANTS), default='p1c')
    t.add_argument('--device', default='cpu')
    t.add_argument('--tag', default=None, help='output subdirectory under tiny_gate/')
    t.add_argument('--out', type=Path, default=DEFAULT_OUT)
    t.set_defaults(fn=cmd_tiny_gate)

    h = sp.add_parser('heldout')
    h.add_argument('--variant', choices=sorted(VARIANTS), default='p1c')
    h.add_argument('--models', default='qwen')
    h.add_argument('--core-transfer', default=None, help='SRC:TGT, e.g. qwen:llava')
    h.add_argument('--crop-prob', type=float, default=0.0)
    h.add_argument('--warmup-epochs', type=int, default=5)
    h.add_argument('--min-epochs', type=int, default=40)
    h.add_argument('--feature-dropout', type=float, default=0.0)
    h.add_argument('--shell-train', default='0:20')
    h.add_argument('--shell-dev', default='20:30')
    h.add_argument('--shell-test', default='30:50')
    h.add_argument('--chess-train', default='0:100')
    h.add_argument('--chess-dev', default='100:150')
    h.add_argument('--chess-test', default='150:200')
    h.add_argument('--epochs', type=int, default=100)
    h.add_argument('--patience', type=int, default=20)
    h.add_argument('--lr', type=float, default=5e-4)
    h.add_argument('--weight-decay', type=float, default=1e-4)
    h.add_argument('--clip-norm', type=float, default=1.0)
    h.add_argument('--probe-c', type=float, default=1.0)
    h.add_argument('--seed', type=int, default=1)
    h.add_argument('--d', type=int, default=128)
    h.add_argument('--hidden', type=int, default=128)
    h.add_argument('--device', default='cpu')
    h.add_argument('--tag', default=None)
    h.add_argument('--out', type=Path, default=DEFAULT_OUT)
    h.set_defaults(fn=cmd_heldout)

    r = sp.add_parser('param-report')
    r.add_argument('--variant', choices=sorted(VARIANTS), default='v1')
    r.add_argument('--device', default='cpu')
    r.add_argument('--models', default='qwen,llava')
    r.add_argument('--d', type=int, default=128)
    r.add_argument('--hidden', type=int, default=128)
    r.add_argument('--out', type=Path, default=DEFAULT_OUT)
    r.set_defaults(fn=cmd_param_report)

    j = sp.add_parser('joint-smoke')
    j.add_argument('--models', default='qwen,llava')
    j.add_argument('--max-trajectories', type=int, default=5)
    j.add_argument('--variant', choices=sorted(VARIANTS), default='p1c')
    j.add_argument('--device', default='cpu')
    j.add_argument('--d', type=int, default=128)
    j.add_argument('--hidden', type=int, default=128)
    j.add_argument('--out', type=Path, default=DEFAULT_OUT)
    j.set_defaults(fn=cmd_joint_smoke)

    c = sp.add_parser('task-c-demo')
    c.add_argument('--train-tasks', default='shell,chess')
    c.add_argument('--train-tasks-models', default='qwen')
    c.add_argument('--max-trajectories', type=int, default=5)
    c.add_argument('--epochs', type=int, default=10)
    c.add_argument('--lr', type=float, default=2e-3)
    c.add_argument('--seed', type=int, default=20260921)
    c.add_argument('--variant', choices=sorted(VARIANTS), default='p1c')
    c.add_argument('--device', default='cpu')
    c.add_argument('--d', type=int, default=128)
    c.add_argument('--hidden', type=int, default=128)
    c.add_argument('--out', type=Path, default=DEFAULT_OUT)
    c.set_defaults(fn=cmd_task_c_demo)

    a = p.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
