"""Universal Semantic Transition Filter (USTF).

One shared transition/observation core across tasks (Shell, Chess, ...) and
backbones (Qwen, LLaVA, ...), trained without task ids, event labels, or
handwritten transition rules. Only two things are backbone-specific: the
`BackboneProjector` (4096 -> d, full linear) and the frozen text-feature
cache it reads from. Everything downstream of that projection -- the visual
change encoder, the pairwise transition scorer, and the observation scorer --
is one set of shared weights, reused unchanged for any task and any number
of candidate states K.

Reuses `Trajectory` / `TextFeatureStore` / `ShellAdapter` / `ChessAdapter`
from `staterev.psf` verbatim: those already match the interface this file
needs (frozen multi-layer hidden, target text, candidate texts, initial
state id, per-step GT state id, trajectory/task/model id), so no new
dataset adapter code is introduced here.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from staterev.psf import Trajectory, TextFeatureStore  # noqa: F401  (re-exported for callers)


class BackboneProjector(nn.Module):
    """Per-backbone map into the shared USTF space.

    `rank=None` is a full linear 4096 -> d (the visual projector). A small
    `rank` gives a low-rank factorization, used for the optional separate
    state-text projector so a second projector per backbone stays inside the
    ~1-2M total parameter budget.
    """

    def __init__(self, hidden_dim: int = 4096, d: int = 128, rank: int | None = None):
        super().__init__()
        self.net = nn.Linear(hidden_dim, d) if rank is None else nn.Sequential(
            nn.Linear(hidden_dim, rank, bias=False), nn.Linear(rank, d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_normalize(raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Standardize a candidate-state text set along the candidate axis.

    Candidate sentences share a template ("the tracked knight is on g1/f3/...")
    so their frozen embeddings are ~0.996 cosine-similar; >90% of each vector
    is the shared sentence component. Centering and scaling over the set
    keeps only what distinguishes the candidates. It is a symmetric function
    of the set, so candidate-order invariance is preserved.
    """
    if raw.shape[0] < 2:
        return raw
    return (raw - raw.mean(0)) / (raw.std(0) + eps)


class USTFCore(nn.Module):
    """Task- and backbone-agnostic transition + observation core.

    Every input here is already projected into the shared d-dim space. The
    core never sees a task id, a backbone id, an event label, or a fixed
    class index -- only vectors and a candidate-state matrix whose size K is
    read off at call time.
    """

    def __init__(self, d: int = 128, hidden: int = 128, transition: str = 'pairwise',
                 obs_match: bool = False, obs_reliability: bool = False):
        super().__init__()
        if transition not in ('pairwise', 'factorized'):
            raise ValueError(f'unknown transition kind {transition!r}')
        self.d = d
        self.transition = transition
        self.obs_match = obs_match
        self.obs_reliability = obs_reliability
        # Softmax-mixture over the 4 fractional-depth taps produced by
        # staterev.psf.tap_layers; separate weights for the change path and
        # the observation path, shared across every backbone/task.
        self.layer_mix_change = nn.Parameter(torch.zeros(4))
        self.layer_mix_obs = nn.Parameter(torch.zeros(4))
        # v_t = f(H_t, H_{t-1}, H_t-H_{t-1}, target)  ->  R^d
        self.change_encoder = nn.Sequential(nn.Linear(4 * d, hidden), nn.GELU(), nn.Linear(hidden, d))
        # score(i,j) = F(v_t, c_i, c_j, c_j-c_i, target)  ->  scalar
        self.transition_scorer = nn.Sequential(nn.Linear(5 * d, hidden), nn.GELU(), nn.Linear(hidden, 1))
        # O_t(j) = G(H_t, c_j, target)  ->  scalar
        self.observation_scorer = nn.Sequential(nn.Linear(3 * d, hidden), nn.GELU(), nn.Linear(hidden, 1))
        if transition == 'factorized':
            # P(stay | v_t, c_i, target), separate from where a move goes.
            self.stay_scorer = nn.Sequential(nn.Linear(3 * d, hidden), nn.GELU(), nn.Linear(hidden, 1))
            # Semantic delta matching: <A v_t, c_j - c_i>.
            self.delta_match = nn.Linear(d, d, bias=False)
        if obs_match:
            # Bilinear hidden/state-text match term added to the MLP score.
            self.obs_bilinear = nn.Linear(d, d, bias=False)
        if obs_reliability:
            # beta_t = softplus(w . [H(p_obs)/log K, max p_obs, H(prior)/log K] + b),
            # initialized to beta = 1 (plain Bayes product).
            self.reliability = nn.Linear(3, 1)
            nn.init.zeros_(self.reliability.weight)
            nn.init.constant_(self.reliability.bias, float(np.log(np.e - 1)))

    @staticmethod
    def _mix(taps: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        # taps: [..., 4, d] -> [..., d]
        w = torch.softmax(weights, dim=0).view(*([1] * (taps.ndim - 2)), 4, 1)
        return (taps * w).sum(-2)

    def visual_change(self, h_t: torch.Tensor, h_prev: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ct = self._mix(h_t, self.layer_mix_change)
        cp = self._mix(h_prev, self.layer_mix_change)
        delta = ct - cp
        return self.change_encoder(torch.cat([ct, cp, delta, target], dim=-1))

    def transition_matrix(self, v_t: torch.Tensor, states: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Row-stochastic T_t(i,j) = P(S_t=j | S_{t-1}=i, v_t) for all K x K pairs."""
        k = states.shape[0]
        ci = states.unsqueeze(1).expand(k, k, self.d)
        cj = states.unsqueeze(0).expand(k, k, self.d)
        delta = cj - ci
        vt = v_t.view(1, 1, self.d).expand(k, k, self.d)
        tgt = target.view(1, 1, self.d).expand(k, k, self.d)
        scores = self.transition_scorer(torch.cat([vt, ci, cj, delta, tgt], dim=-1)).squeeze(-1)
        if self.transition == 'pairwise':
            return torch.softmax(scores, dim=-1)
        # factorized: T(i,i) = p_stay(i); T(i,j != i) = (1 - p_stay(i)) * softmax_{j != i}(move score)
        if k == 1:
            return torch.ones(1, 1, device=states.device)
        scores = scores + (delta * self.delta_match(v_t)).sum(-1) / self.d ** 0.5
        eye = torch.eye(k, dtype=torch.bool, device=states.device)
        move = torch.softmax(scores.masked_fill(eye, float('-inf')), dim=-1)
        p_stay = torch.sigmoid(self.stay_scorer(torch.cat([v_t.expand(k, self.d), states, target.expand(k, self.d)], -1))).squeeze(-1)
        return torch.diag(p_stay) + (1 - p_stay).unsqueeze(-1) * move

    def observation_logits(self, h_t: torch.Tensor, states: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        k = states.shape[0]
        hm = self._mix(h_t, self.layer_mix_obs)
        ht = hm.view(1, self.d).expand(k, self.d)
        tgt = target.view(1, self.d).expand(k, self.d)
        logits = self.observation_scorer(torch.cat([ht, states, tgt], dim=-1)).squeeze(-1)
        if self.obs_match:
            logits = logits + states @ self.obs_bilinear(hm) / self.d ** 0.5
        return logits

    def fuse(self, prior: torch.Tensor, obs_logits: torch.Tensor) -> torch.Tensor:
        """Bayes-style correction: posterior ∝ prior * p_obs^beta (log-space sum)."""
        log_obs = F.log_softmax(obs_logits, dim=-1)
        beta = 1.0
        if self.obs_reliability and prior.shape[0] > 1:
            log_k = float(np.log(prior.shape[0]))
            p_obs = log_obs.exp()
            feats = torch.stack([-(p_obs * log_obs).sum() / log_k, p_obs.max(),
                                 -(prior * torch.log(prior.clamp_min(1e-8))).sum() / log_k])
            beta = F.softplus(self.reliability(feats)).squeeze()
        return F.softmax(torch.log(prior.clamp_min(1e-8)) + beta * log_obs.clamp_min(np.log(1e-8)), dim=-1)


class USTF(nn.Module):
    """Frozen-VLM-facing wrapper: per-backbone projector + shared USTFCore."""

    def __init__(self, backbones=('qwen',), hidden_dim: int = 4096, d: int = 128, hidden: int = 128, text_stores=None,
                 text_norm: bool = False, text_projector_rank: int | None = None, transition: str = 'pairwise',
                 obs_match: bool = False, obs_reliability: bool = False, feature_dropout: float = 0.0,
                 hidden_norm: bool = False):
        """hidden_norm: non-affine LayerNorm on each raw hidden tap before projection.
        feature_dropout: dropout on projected visual taps (training only).
        text_norm: set-normalize candidate text features (see set_normalize).
        text_projector_rank: None shares the visual projector for text (v1);
        an int adds a separate low-rank state-text projector per backbone.
        transition / obs_match / obs_reliability: see USTFCore.
        """
        super().__init__()
        self.d = d
        self.hidden_dim = hidden_dim
        self.config = dict(hidden=hidden, text_norm=text_norm, text_projector_rank=text_projector_rank,
                           transition=transition, obs_match=obs_match, obs_reliability=obs_reliability,
                           feature_dropout=feature_dropout, hidden_norm=hidden_norm)
        self.hidden_norm = hidden_norm
        self.feature_dropout = nn.Dropout(feature_dropout)
        self.text_norm = text_norm
        self.text_projector_rank = text_projector_rank
        self.projectors = nn.ModuleDict({b: BackboneProjector(hidden_dim, d) for b in backbones})
        self.text_projectors = nn.ModuleDict(
            {b: BackboneProjector(hidden_dim, d, text_projector_rank) for b in backbones}
            if text_projector_rank is not None else {})
        self.core = USTFCore(d, hidden, transition=transition, obs_match=obs_match, obs_reliability=obs_reliability)
        self.text_stores = text_stores or {b: TextFeatureStore(dim=hidden_dim) for b in backbones}
        self.use_observation = True

    def add_backbone(self, name: str, hidden_dim: int = 4096, text_store: TextFeatureStore | None = None):
        """Register a new backbone projector without touching the shared core."""
        if name in self.projectors:
            return
        device = next(self.parameters()).device
        self.projectors[name] = BackboneProjector(hidden_dim, self.d).to(device)
        if self.text_projector_rank is not None:
            self.text_projectors[name] = BackboneProjector(hidden_dim, self.d, self.text_projector_rank).to(device)
        if text_store is not None:
            self.text_stores[name] = text_store
        elif name not in self.text_stores:
            raise ValueError(f'no text feature store registered for new backbone {name!r}')

    def encode_texts(self, texts, model: str, device, candidate_set: bool = False) -> torch.Tensor:
        store = self.text_stores[model]
        raw = torch.stack([store.get(t).to(device) for t in texts]).float()
        if candidate_set and self.text_norm:
            raw = set_normalize(raw)
        proj = self.text_projectors[model] if self.text_projector_rank is not None else self.projectors[model]
        return proj(raw)

    def forward_trajectory(self, tr: Trajectory, permutation: torch.Tensor | None = None) -> dict:
        """Free-rollout forward pass over one trajectory.

        `permutation` is an optional length-K index tensor used only by the
        candidate-order invariance check: candidate texts, GT state ids, and
        the initial state id are relabeled consistently before the forward
        pass, so a correct implementation must assign the identical
        probability to each underlying candidate regardless of where that
        candidate happens to sit in the input list.
        """
        dev = next(self.parameters()).device
        candidate_texts = list(tr.candidate_texts)
        state_ids = tr.state_ids
        initial_state_id = tr.initial_state_id
        if permutation is not None:
            perm = [int(x) for x in permutation.tolist()]
            candidate_texts = [candidate_texts[i] for i in perm]
            inverse = {old: new for new, old in enumerate(perm)}
            state_ids = torch.tensor([inverse[int(s)] for s in state_ids])
            initial_state_id = inverse[int(initial_state_id)]

        raw = tr.hidden.to(dev).float()
        if self.hidden_norm:
            raw = F.layer_norm(raw, raw.shape[-1:])
        h = self.feature_dropout(self.projectors[tr.model_name](raw))  # [T, 4, d]
        target = self.encode_texts([tr.target_text], tr.model_name, dev)[0]
        states = self.encode_texts(candidate_texts, tr.model_name, dev, candidate_set=True)  # [K, d]
        k = states.shape[0]

        belief = torch.zeros(k, device=dev)
        belief[initial_state_id] = 1.0
        prev_h = None
        transition_matrices, obs_logits_seq, priors, posteriors = [], [], [], []
        for ht in h:
            hp = ht if prev_h is None else prev_h
            v_t = self.core.visual_change(ht, hp, target)
            t_t = self.core.transition_matrix(v_t, states, target)
            prior = belief @ t_t
            obs_logits = self.core.observation_logits(ht, states, target)
            # Bayes-filter-style multiplicative correction in log space
            # (equivalent to a normalized elementwise product): deliberately
            # not a fixed 0.5-weighted average of prior and observation.
            # `use_observation=False` is the prior-only (transition-only) ablation.
            post = self.core.fuse(prior, obs_logits) if self.use_observation else prior
            transition_matrices.append(t_t)
            obs_logits_seq.append(obs_logits)
            priors.append(prior)
            posteriors.append(post)
            belief = post
            prev_h = ht
        return {
            'transition_matrices': torch.stack(transition_matrices),  # [T,K,K]
            'obs_logits': torch.stack(obs_logits_seq),  # [T,K]
            'prior': torch.stack(priors),  # [T,K]
            'posterior': torch.stack(posteriors),  # [T,K]
            'state_embeddings': states,
            'state_ids': state_ids.to(dev),
            'initial_state_id': initial_state_id,
        }

    def parameter_counts(self, vlm_param_counts=(7e9, 8e9)) -> dict:
        core = self.core

        def n(mod):
            return sum(p.numel() for p in mod.parameters()) if mod is not None else 0

        proj = {k: n(v) for k, v in self.projectors.items()}
        shared_text = self.text_projector_rank is None
        shared_core = n(core)
        total = sum(p.numel() for p in self.parameters())
        return {
            'per_backbone_projector': proj,
            # When shared, the state-text projector is literally the visual
            # projector module; it is still listed because the spec asks for it.
            'state_text_projector': dict(proj) if shared_text else {k: n(v) for k, v in self.text_projectors.items()},
            'state_text_projector_shared_with_visual_projector': shared_text,
            'visual_change_encoder': n(core.change_encoder),
            'transition_scorer': n(core.transition_scorer) + n(getattr(core, 'stay_scorer', None)) + n(getattr(core, 'delta_match', None)),
            'observation_scorer': n(core.observation_scorer) + n(getattr(core, 'obs_bilinear', None)) + n(getattr(core, 'reliability', None)),
            'layer_mixture_weights': int(core.layer_mix_change.numel() + core.layer_mix_obs.numel()),
            'shared_core_total': shared_core,
            'total_trainable': total,
            'fraction_of_vlm': {str(int(c)): total / c for c in vlm_param_counts},
        }


def ustf_loss(model: USTF, tr: Trajectory, permutation: torch.Tensor | None = None,
              w_observation: float = 0.5, w_rollout: float = 1.0):
    """L = L_transition + w_observation * L_observation + w_rollout * L_rollout.

    L_transition and L_observation are teacher-forced against GT state ids
    (they read S_{t-1}/S_t directly, never the model's own recursive
    belief). L_rollout is read off `posterior`, which is produced by a fully
    free recursion (belief_t = posterior_{t-1}, no GT reinjection) -- so it
    is the only term that supervises actual deployed rollout behaviour.
    """
    o = model.forward_trajectory(tr, permutation=permutation)
    dev = o['posterior'].device
    y = o['state_ids'].to(dev)
    prev = torch.cat([torch.tensor([o['initial_state_id']], device=dev), y[:-1]])
    idx = torch.arange(len(y), device=dev)
    trans_ll = o['transition_matrices'][idx, prev, y]
    l_transition = -torch.log(trans_ll.clamp_min(1e-8)).mean()
    l_observation = F.cross_entropy(o['obs_logits'], y)
    l_rollout = -torch.log(o['posterior'][idx, y].clamp_min(1e-8)).mean()
    loss = l_transition + w_observation * l_observation + w_rollout * l_rollout
    return loss, {'transition': float(l_transition.detach()), 'observation': float(l_observation.detach()),
                  'rollout': float(l_rollout.detach())}


def evaluate_ustf(model: USTF, data) -> pd.DataFrame:
    rows = []
    model.eval()
    with torch.no_grad():
        for tr in data:
            o = model.forward_trajectory(tr)
            y = o['state_ids'].cpu()
            prev = torch.cat([torch.tensor([o['initial_state_id']]), y[:-1]])
            t = o['transition_matrices'].cpu()
            trans_pred = t[torch.arange(len(y)), prev].argmax(-1)
            obs_pred = o['obs_logits'].argmax(-1).cpu()
            rollout_pred = o['posterior'].argmax(-1).cpu()
            for i in range(len(y)):
                rows.append({
                    'task': tr.task_name, 'model': tr.model_name, 'trajectory_id': tr.trajectory_id,
                    't': i + 1, 'gt': int(y[i]), 'num_candidates': int(o['state_embeddings'].shape[0]),
                    'initial_state': int(o['initial_state_id']), 'state_changed': int(y[i] != prev[i]),
                    'transition_pred': int(trans_pred[i]), 'transition_correct': int(trans_pred[i] == y[i]),
                    'observation_pred': int(obs_pred[i]), 'observation_correct': int(obs_pred[i] == y[i]),
                    'rollout_pred': int(rollout_pred[i]), 'rollout_correct': int(rollout_pred[i] == y[i]),
                })
    return pd.DataFrame(rows)


def permutation_invariance_check(model: USTF, tr: Trajectory, permutation: torch.Tensor, atol: float = 1e-4) -> bool:
    model.eval()
    with torch.no_grad():
        o_id = model.forward_trajectory(tr)
        o_perm = model.forward_trajectory(tr, permutation=permutation)
    inverse = torch.argsort(permutation)
    ok_post = torch.allclose(o_id['posterior'], o_perm['posterior'][:, inverse], atol=atol)
    ok_prior = torch.allclose(o_id['prior'], o_perm['prior'][:, inverse], atol=atol)
    return bool(ok_post and ok_prior)


def crop_trajectory(tr: Trajectory, start: int) -> Trajectory:
    """Suffix of `tr` from step `start`, initialized at the GT state just before it."""
    if start <= 0:
        return tr
    return Trajectory(tr.model_name, tr.task_name, tr.trajectory_id, tr.hidden[start:], tr.target_text,
                      tr.candidate_texts, int(tr.state_ids[start - 1]), tr.state_ids[start:],
                      None if tr.native_ids is None else tr.native_ids[start:], tr.schema_fallback)


def train_ustf(model: USTF, train_data, dev_data, epochs: int = 30, lr: float = 1e-3, seed: int = 17,
               w_observation: float = 0.5, w_rollout: float = 1.0, permute_candidates: bool = True,
               patience: int = 8, weight_decay: float = 1e-4, clip_norm: float = 1.0,
               crop_prob: float = 0.0, warmup_epochs: int = 0, min_epochs: int = 0) -> pd.DataFrame:
    """Episodic, task-balanced training. No event field is ever read.

    Each episode independently draws a fresh candidate-order permutation
    (when `permute_candidates`); target entity and trajectory identity are
    already free per-Trajectory fields (see staterev.psf.Trajectory), so no
    fixed class index or task id ever reaches USTFCore. With probability
    `crop_prob` a training trajectory is replaced by a random suffix started
    from its GT state (varies initial states and trajectory lengths).

    `warmup_epochs` linearly ramps the learning rate from lr/warmup_epochs
    to lr. Early stopping is not armed before `min_epochs`: training often
    sits on an initial "always predict the most frequent state" plateau for
    tens of epochs before the transition scorer starts using v_t, and
    patience would otherwise stop it there.
    """
    torch.manual_seed(seed)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)
    tasks = sorted({x.task_name for x in train_data})
    by_task = {t: [x for x in train_data if x.task_name == t] for t in tasks}
    history = []
    best = None
    bad = 0
    for ep in range(epochs):
        if warmup_epochs:
            for group in opt.param_groups:
                group['lr'] = lr * min(1.0, (ep + 1) / warmup_epochs)
        model.train()
        n = max(len(v) for v in by_task.values())
        episode = []
        for i in range(n):
            order = tasks.copy()
            rng.shuffle(order)
            episode += [by_task[t][i % len(by_task[t])] for t in order]
        losses = []
        for tr in episode:
            if crop_prob and len(tr.state_ids) > 1 and rng.random() < crop_prob:
                tr = crop_trajectory(tr, int(rng.integers(1, len(tr.state_ids))))
            perm = torch.from_numpy(rng.permutation(len(tr.candidate_texts))) if permute_candidates else None
            opt.zero_grad()
            loss, _ = ustf_loss(model, tr, permutation=perm, w_observation=w_observation, w_rollout=w_rollout)
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, clip_norm)
            opt.step()
            losses.append(float(loss.detach()))
        dv = evaluate_ustf(model, dev_data)
        metric = float(dv.rollout_correct.mean()) if len(dv) else 0.0
        history.append({'epoch': ep + 1, 'train_loss': float(np.mean(losses)), 'dev_rollout_accuracy': metric})
        if best is None or metric > best[0]:
            best = (metric, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            bad = 0
        else:
            bad += 1
        if bad >= patience and ep + 1 >= min_epochs:
            break
    if best is not None:
        model.load_state_dict(best[1])
    return pd.DataFrame(history)


def adapt_to_new_task(model: USTF, target_data, mode: str = 'zero-shot', few_shot_n: int | None = None,
                       epochs: int = 10, lr: float = 1e-3, seed: int = 17):
    """The A+B -> C interface: `model` is a USTF trained on source tasks.

    `mode` in {'zero-shot', 'few-shot', 'from-scratch'}. In every mode,
    USTFCore (change_encoder / transition_scorer / observation_scorer) is
    the exact same module object with the exact same parameter count --
    task C is never given its own output head, and any new candidate count
    K is handled purely by the pairwise scorer's existing K-agnostic shape.
    Only 'few-shot' calibrates parameters, and only the per-backbone
    projector(s) (plus, for a brand-new backbone id, a freshly initialized
    BackboneProjector registered via `add_backbone`).
    """
    if mode not in ('zero-shot', 'few-shot', 'from-scratch'):
        raise ValueError(f'unknown mode {mode!r}')
    core_params_before = sum(p.numel() for p in model.core.parameters())

    if mode == 'from-scratch':
        backbones = tuple(sorted({x.model_name for x in target_data}))
        fresh = USTF(backbones=backbones, hidden_dim=model.hidden_dim, d=model.d, text_stores=model.text_stores,
                     **model.config).to(next(model.parameters()).device)
        hist = train_ustf(fresh, target_data, target_data, epochs=epochs, lr=lr, seed=seed)
        assert sum(p.numel() for p in fresh.core.parameters()) == core_params_before
        return fresh, hist

    for backbone in sorted({x.model_name for x in target_data}):
        if backbone not in model.projectors:
            model.add_backbone(backbone)

    if mode == 'zero-shot':
        assert sum(p.numel() for p in model.core.parameters()) == core_params_before
        return model, pd.DataFrame()

    if not few_shot_n:
        raise ValueError('mode=few-shot requires few_shot_n')
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(('projectors.', 'text_projectors.'))
    hist = train_ustf(model, target_data[:few_shot_n], target_data, epochs=epochs, lr=lr, seed=seed)
    for p in model.parameters():
        p.requires_grad = True
    assert sum(p.numel() for p in model.core.parameters()) == core_params_before
    return model, hist


def run_tiny_gate(model: USTF, shell_data, chess_data, out_dir: Path, epochs: int = 40, lr: float = 5e-4,
                   seed: int = 20260921, permute_candidates: bool = True, patience: int = 8,
                   weight_decay: float = 0.0, clip_norm: float = 1.0) -> tuple[dict, USTF]:
    """Overfit gate: fit and evaluate on the same tiny discovery set.

    This is a learnability/sanity check, not a held-out generalization
    claim -- matching the "tiny gate" role described for USTF v1.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    data = list(shell_data) + list(chess_data)
    assert data, 'tiny gate requires at least one trajectory'

    model.eval()
    with torch.no_grad():
        pre = evaluate_ustf(model, data)

    hist = train_ustf(model, data, data, epochs=epochs, lr=lr, seed=seed, permute_candidates=permute_candidates,
                       patience=patience, weight_decay=weight_decay, clip_norm=clip_norm)
    post = evaluate_ustf(model, data)

    def acc(df, col):
        return {task: float(g[col].mean()) for task, g in df.groupby('task')}

    rng = np.random.default_rng(seed + 1)
    perm_results = {}
    for tr in data:
        perm = torch.from_numpy(rng.permutation(len(tr.candidate_texts)))
        perm_results[f'{tr.task_name}_{tr.trajectory_id}'] = permutation_invariance_check(model, tr, perm)

    pre_obs = acc(pre, 'observation_correct')
    post_obs = acc(post, 'observation_correct')
    post_trans = acc(post, 'transition_correct')
    post_roll = acc(post, 'rollout_correct')

    report = {
        'trajectories': {t: int((post.task == t).sum() > 0) and int(post[post.task == t].trajectory_id.nunique()) for t in post.task.unique()},
        'epochs_run': int(len(hist)),
        'pre_training_observation_accuracy': pre_obs,
        'post_training_transition_accuracy': post_trans,
        'post_training_rollout_accuracy': post_roll,
        'post_training_observation_accuracy': post_obs,
        'permutation_invariant_fraction': float(np.mean(list(perm_results.values()))) if perm_results else 0.0,
        'permutation_failures': [k for k, v in perm_results.items() if not v],
        'parameter_counts': model.parameter_counts(),
        'history_tail': hist.tail(5).to_dict('records'),
    }
    transition_pass = all(v >= 0.95 for v in post_trans.values())
    rollout_pass = all(v >= 0.95 for v in post_roll.values())
    observation_pass = all(post_obs[t] >= 0.5 and post_obs[t] - pre_obs.get(t, 0.0) >= 0.1 for t in post_obs)
    permutation_pass = report['permutation_invariant_fraction'] == 1.0
    report['gate'] = {
        'transition_pass': transition_pass, 'rollout_pass': rollout_pass,
        'observation_pass': observation_pass, 'permutation_pass': permutation_pass,
        'overall_pass': bool(transition_pass and rollout_pass and observation_pass and permutation_pass),
    }
    (out_dir / 'tiny_gate_report.json').write_text(json.dumps(report, indent=2) + '\n')
    post.to_csv(out_dir / 'tiny_gate_predictions.csv', index=False)
    hist.to_csv(out_dir / 'tiny_gate_history.csv', index=False)
    torch.save(model.state_dict(), out_dir / 'tiny_gate_model.pt')
    return report, model
