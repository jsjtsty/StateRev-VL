from pathlib import Path

import numpy as np
import torch

from staterev.psf import Trajectory, TextFeatureStore, ShellAdapter
from staterev.ustf import (
    USTF, USTFCore, BackboneProjector, ustf_loss, evaluate_ustf, train_ustf,
    permutation_invariance_check, adapt_to_new_task, run_tiny_gate,
)

ROOT = Path(__file__).resolve().parents[1]


def fake(model='qwen', task='shell', n=4, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, 4, 4096, generator=g)
    y = torch.tensor([0, 0, 1, 1][:n]) if k >= 2 else torch.zeros(n, dtype=torch.long)
    return Trajectory(model, task, f'x{seed}', h, 'tracked object', [f'state {i}' for i in range(k)], 0, y)


def test_no_manual_event_interface():
    tr = fake()
    assert not hasattr(tr, 'event') and not hasattr(tr, 'gt_event')


def test_shapes_and_loss_finite():
    m = USTF(('qwen',))
    tr = fake(n=4, k=3)
    o = m.forward_trajectory(tr)
    assert o['transition_matrices'].shape == (4, 3, 3)
    assert o['obs_logits'].shape == (4, 3) and o['posterior'].shape == (4, 3) and o['prior'].shape == (4, 3)
    assert torch.allclose(o['transition_matrices'].sum(-1), torch.ones(4, 3), atol=1e-5)
    assert torch.allclose(o['posterior'].sum(-1), torch.ones(4), atol=1e-5)
    loss, parts = ustf_loss(m, tr)
    assert torch.isfinite(loss) and set(parts) == {'transition', 'observation', 'rollout'}


def test_cross_backbone_shared_core_separate_projectors():
    m = USTF(('qwen', 'llava'))
    assert m.projectors['qwen'] is not m.projectors['llava']
    oq = m.forward_trajectory(fake('qwen'))
    ol = m.forward_trajectory(fake('llava'))
    assert oq['posterior'].shape == ol['posterior'].shape
    core_params = sum(p.numel() for p in m.core.parameters())
    m2 = USTF(('qwen',))
    assert sum(p.numel() for p in m2.core.parameters()) == core_params


def test_variable_k_joint_task_forward():
    m = USTF(('qwen',))
    o_shell = m.forward_trajectory(fake('qwen', 'shell', n=5, k=3))
    o_chess = m.forward_trajectory(fake('qwen', 'chess', n=6, k=65))
    assert o_shell['posterior'].shape == (5, 3)
    assert o_chess['posterior'].shape == (6, 65)


def test_candidate_permutation_invariance():
    m = USTF(('qwen',))
    tr = fake(n=6, k=8)
    perm = torch.from_numpy(np.random.default_rng(3).permutation(8))
    assert permutation_invariance_check(m, tr, perm)


V2 = dict(text_norm=True, text_projector_rank=64, transition='factorized', obs_match=True, obs_reliability=True)


def test_v2_factorized_rows_stochastic_and_variable_k():
    m = USTF(('qwen',), **V2)
    for k in (1, 2, 3, 65):
        o = m.forward_trajectory(fake(n=3, k=k))
        assert o['transition_matrices'].shape == (3, k, k)
        assert torch.allclose(o['transition_matrices'].sum(-1), torch.ones(3, k), atol=1e-5)
        assert torch.allclose(o['posterior'].sum(-1), torch.ones(3), atol=1e-5)
    loss, _ = ustf_loss(m, fake(n=4, k=5))
    loss.backward()
    assert m.core.stay_scorer[0].weight.grad is not None and m.core.reliability.weight.grad is not None


def test_v2_candidate_permutation_invariance():
    m = USTF(('qwen',), **V2)
    tr = fake(n=6, k=8)
    perm = torch.from_numpy(np.random.default_rng(3).permutation(8))
    assert permutation_invariance_check(m, tr, perm)


def test_set_normalize_removes_shared_component():
    from staterev.ustf import set_normalize
    base = torch.randn(1, 4096)
    raw = base + 0.01 * torch.randn(5, 4096)
    z = set_normalize(raw)
    assert torch.allclose(z.mean(0), torch.zeros(4096), atol=1e-3)
    assert torch.allclose(set_normalize(raw[[2, 0, 4, 1, 3]]), z[[2, 0, 4, 1, 3]], atol=1e-3)


def test_reliability_initialized_to_plain_bayes_product():
    torch.manual_seed(0)
    a = USTF(('qwen',), obs_reliability=False)
    b = USTF(('qwen',), obs_reliability=True)
    b.load_state_dict(a.state_dict(), strict=False)
    tr = fake(n=4, k=3)
    assert torch.allclose(a.forward_trajectory(tr)['posterior'], b.forward_trajectory(tr)['posterior'], atol=1e-5)


def test_v2_parameter_budget_two_backbones():
    counts = USTF(('qwen', 'llava'), **V2).parameter_counts()
    assert counts['state_text_projector_shared_with_visual_projector'] is False
    assert counts['total_trainable'] < 2_000_000


def test_crop_trajectory_starts_from_gt_state():
    from staterev.ustf import crop_trajectory
    tr = fake(n=4, k=3)  # states 0,0,1,1
    c = crop_trajectory(tr, 2)
    assert c.initial_state_id == 0 and c.state_ids.tolist() == [1, 1] and c.hidden.shape[0] == 2
    assert crop_trajectory(tr, 0) is tr
    m = USTF(('qwen',), feature_dropout=0.3)
    hist = train_ustf(m, [tr], [tr], epochs=2, crop_prob=1.0)
    assert len(hist) == 2


def test_parameter_budget_small_and_shared():
    m = USTF(('qwen', 'llava'))
    counts = m.parameter_counts()
    assert counts['total_trainable'] < 2_000_000
    assert counts['shared_core_total'] < 250_000
    assert counts['fraction_of_vlm']['7000000000'] < 1e-3
    assert set(counts['per_backbone_projector']) == {'qwen', 'llava'}


def test_task_c_interface_never_grows_core():
    torch.manual_seed(0)
    m = USTF(('qwen',))
    core_n = sum(p.numel() for p in m.core.parameters())
    synthetic_c = [fake('qwen', 'synthetic_c', n=3, k=4, seed=i) for i in range(12)]

    m_zero, hist_zero = adapt_to_new_task(m, synthetic_c, mode='zero-shot')
    assert hist_zero.empty and sum(p.numel() for p in m_zero.core.parameters()) == core_n

    for n_shot in (1, 5, 10):
        m_few, hist_few = adapt_to_new_task(m, synthetic_c, mode='few-shot', few_shot_n=n_shot, epochs=2)
        assert sum(p.numel() for p in m_few.core.parameters()) == core_n
        assert all(p.requires_grad for p in m_few.parameters())

    m_scratch, hist_scratch = adapt_to_new_task(m, synthetic_c, mode='from-scratch', epochs=2)
    assert sum(p.numel() for p in m_scratch.core.parameters()) == core_n
    assert m_scratch is not m


def test_task_c_interface_registers_new_backbone_without_touching_core():
    m = USTF(('qwen',))
    core_before = {k: v.clone() for k, v in m.core.state_dict().items()}
    new_backbone_data = [fake('brand_new_backbone', 'synthetic_c', n=3, k=4, seed=i) for i in range(3)]
    store = {'brand_new_backbone': TextFeatureStore(dim=4096)}
    m.text_stores.update(store)
    m_adapted, _ = adapt_to_new_task(m, new_backbone_data, mode='zero-shot')
    assert 'brand_new_backbone' in m_adapted.projectors
    for k, v in core_before.items():
        assert torch.equal(v, m_adapted.core.state_dict()[k])


def test_strict_text_cache_fails_closed():
    try:
        TextFeatureStore('/definitely/missing/ustf_text.npz', strict=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('strict text cache must fail closed')


def test_train_ustf_overfits_tiny_synthetic_trajectory():
    torch.manual_seed(0)
    m = USTF(('qwen',))
    data = [fake('qwen', 'shell', n=4, k=3, seed=i) for i in range(3)]
    hist = train_ustf(m, data, data, epochs=25, lr=5e-3, seed=1)
    ev = evaluate_ustf(m, data)
    assert hist.dev_rollout_accuracy.iloc[-1] >= hist.dev_rollout_accuracy.iloc[0]
    assert ev.rollout_correct.mean() > 0.5


def test_tiny_gate_runs_end_to_end_on_synthetic_data(tmp_path):
    torch.manual_seed(0)
    m = USTF(('qwen',))
    shell = [fake('qwen', 'shell', n=4, k=3, seed=i) for i in range(3)]
    chess = [fake('qwen', 'chess', n=3, k=6, seed=100 + i) for i in range(2)]
    report, _ = run_tiny_gate(m, shell, chess, tmp_path, epochs=15, lr=5e-3)
    assert set(report['gate']) == {'transition_pass', 'rollout_pass', 'observation_pass', 'permutation_pass', 'overall_pass'}
    assert (tmp_path / 'tiny_gate_report.json').exists()


def test_real_cache_schema_qwen_llava_if_present():
    hidden_qwen = ROOT / 'outputs/vetbench/hidden_state_probe/hidden_states.npz'
    if not hidden_qwen.exists():
        return
    tr = ShellAdapter('qwen').load(1)[0]
    assert tr.hidden.ndim == 3 and tr.hidden.shape[1] == 4 and tr.hidden.shape[2] == 4096
