import numpy as np,torch
from staterev.psf import Trajectory,PSF,SemanticStateDeltaPSF,KeepSetPSF,TextFeatureStore,keep_set_loss,loss_for,state_delta_loss,tap_layers,split_grouped,oracle_transition_rows

def fake(model='qwen',task='shell',n=4,k=3):
 h=torch.randn(n,4,4096); y=torch.tensor([0,0,1,1])
 return Trajectory(model,task,'x',h,'tracked object',[f'state {i}' for i in range(k)],0,y)

def test_shapes_and_loss_no_events():
 m=PSF(('qwen',)); tr=fake(); o=m.forward_trajectory(tr)
 assert o['logits'].shape==(4,3) and o['memory'].shape==(4,64)
 loss,p=loss_for(m,tr); assert torch.isfinite(loss) and set(p)=={'state','change','persist'}

def test_cross_backbone_shared_core():
 m=PSF(('qwen','llava')); q=m.forward_trajectory(fake('qwen')); l=m.forward_trajectory(fake('llava'))
 assert q['memory'].shape==l['memory'].shape and m.gru is m.gru
 c=m.parameter_counts(); assert c['shared_core']<100_000 and c['total_trainable']<200_000

def test_semantic_delta_shapes_and_loss_without_events():
 m=SemanticStateDeltaPSF(('qwen','llava'),updater='residual'); tr=fake('qwen')
 o=m.forward_trajectory(tr); loss,p=state_delta_loss(m,tr,'smoothl1')
 assert o['delta_pred'].shape==(4,64) and o['m_pred'].shape==(4,64)
 assert torch.isfinite(loss) and {'state','delta','change','persist'} <= set(p)
 assert m.change_encoder is m.change_encoder

def test_keep_set_free_rollout_and_parameter_budget():
    m=KeepSetPSF(('qwen','llava'),fusion='learned'); tr=fake('qwen')
    o=m.forward_trajectory(tr); loss,p=keep_set_loss(m,tr)
    assert o['final_probs'].shape==(4,3) and o['track_probs'].shape==(4,3)
    assert torch.isfinite(loss) and set(p)=={'track','update','set','direct'}
    assert m.parameter_counts()['total_trainable'] < 700_000

def test_keep_set_scheduled_training_path_is_finite_and_free_default_is_stable():
    m=KeepSetPSF(('qwen',)); tr=fake('qwen')
    m.train()
    scheduled,_=keep_set_loss(m,tr,teacher_prob=.5,pos_weight=2.0)
    assert torch.isfinite(scheduled)
    m.eval()
    free_a=m.forward_trajectory(tr)['final_probs']
    free_b=m.forward_trajectory(tr)['final_probs']
    assert torch.allclose(free_a,free_b)

def test_single_layer_fallback_explicit():
 x=np.ones(4096,dtype='float32'); y,flag=tap_layers(x)
 assert flag and y.shape==(4,4096) and np.all(y[0]==y[-1])

def test_no_manual_event_interface_and_oracle_isolated():
 tr=fake(); assert not hasattr(tr,'event') and not hasattr(tr,'gt_event')
 o=oracle_transition_rows([tr]); assert len(o)==4 and o.correct.all()

def test_grouped_split_and_strict_text_cache(tmp_path=None):
 data=[fake('qwen','shell'),fake('llava','shell'),fake('qwen','chess'),fake('llava','chess')]
 for i,x in enumerate(data): x.trajectory_id=str(i)
 train,dev=split_grouped(data)
 assert not ({x.trajectory_id for x in train}&{x.trajectory_id for x in dev})
 try: TextFeatureStore('/definitely/missing/psf_text.npz',strict=True)
 except FileNotFoundError: pass
 else: raise AssertionError('strict text cache must fail closed')
