"""Independent parity check against the pinned original projected-embedding core.

This module does not launch a run. Import it in the verification driver.
score_fn(embeddings) must return a differentiable scalar/batch CLIP cosine,
with no logit scaling. embeddings has shape [1, optimized_tokens, width].
The task adapts the allowed-ID mask/model/target/16-slot template only.
"""
from pathlib import Path
import ast
import contextlib
import hashlib
import json
import torch

COMMIT = '21eb35f16697529388a43bf1504089eccc6bc432'


def original_nn_project(source_dir=None):
    source_dir = Path(source_dir or Path(__file__).parent / 'original_source')
    manifest = json.loads((source_dir / 'manifest.json').read_text())
    assert manifest['commit'] == COMMIT
    record = next(r for r in manifest['files'] if r['file'] == 'optim_utils.py')
    raw = (source_dir / 'optim_utils.py').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['sha256']
    tree = ast.parse(raw.decode('utf8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'nn_project')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_dir/'optim_utils.py'), 'exec'), namespace)
    return namespace['nn_project']


@contextlib.contextmanager
def capture_adamw_steps(max_steps=5):
    """Capture actual minimal-script updates; parent adds IDs and trace scores."""
    records = []
    original_step = torch.optim.AdamW.step

    def capture(optimizer, *args, **kwargs):
        params = [p for group in optimizer.param_groups for p in group['params']]
        keep = len(records) < max_steps and len(params) == 1 and params[0].grad is not None
        if keep:
            parameter = params[0]
            before = parameter.detach().clone()
            gradient = parameter.grad.detach().clone()
        result = original_step(optimizer, *args, **kwargs)
        if keep:
            state = optimizer.state[parameter]
            records.append(dict(parameter_before=before, gradient=gradient,
                parameter=parameter.detach().clone(),
                exp_avg=state['exp_avg'].detach().clone(),
                exp_avg_sq=state['exp_avg_sq'].detach().clone(),
                optimizer_state_step=int(state['step'].item())))
        return result

    torch.optim.AdamW.step = capture
    try:
        yield records
    finally:
        torch.optim.AdamW.step = original_step


def reference_trajectory(embedding_layer, initial_embeds, score_fn, allowed_ids,
                         steps=5, lr=0.1, weight_decay=0.1, source_dir=None):
    """Original optim_utils.py L174-218, using the unmodified original projector."""
    project = original_nn_project(source_dir)
    parameter = initial_embeds.detach().clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([parameter], lr=lr, weight_decay=weight_decay)
    allowed_ids = set(int(i) for i in allowed_ids)
    forbidden_set = set(range(embedding_layer.num_embeddings)) - allowed_ids
    forbidden_mask = torch.tensor([i in forbidden_set for i in range(embedding_layer.num_embeddings)],
                                  dtype=torch.bool, device=parameter.device)
    records = []
    for step in range(steps):
        # Original nn_project normalizes queries/table for the nearest neighbor,
        # then returns the original, UNnormalized token embeddings.
        projected, ids = project(parameter, embedding_layer,
            forbidden_mask=forbidden_mask, forbidden_set=forbidden_set, english_mask=None)
        with torch.no_grad():
            score = float(score_fn(projected).mean().item())
        # Keep the continuous parameter separate from the projection point.
        temporary = parameter.detach().clone()
        temporary.data = projected.data
        temporary.requires_grad = True
        loss = 1 - score_fn(temporary).mean()
        parameter.grad, = torch.autograd.grad(loss, [temporary])
        record = dict(step=step, projected_ids=ids.detach().clone(), score=score,
                      loss=float(loss.detach().item()),
                      parameter_before=parameter.detach().clone(),
                      gradient=parameter.grad.detach().clone())
        optimizer.step()
        optimizer.zero_grad()
        state = optimizer.state[parameter]
        record.update(parameter=parameter.detach().clone(),
            exp_avg=state['exp_avg'].detach().clone(),
            exp_avg_sq=state['exp_avg_sq'].detach().clone(),
            optimizer_state_step=int(state['step'].item()))
        records.append(record)
    return records


def compare_trajectories(actual, reference, atol=1e-6, rtol=1e-5):
    assert len(actual) == len(reference) and actual
    checks = []
    for step, (got, expected) in enumerate(zip(actual, reference)):
        item = dict(step=step, projection_ids_exact=torch.equal(
            torch.as_tensor(got['projected_ids'], device=expected['projected_ids'].device).reshape_as(expected['projected_ids']),
            expected['projected_ids']), score_abs_error=abs(float(got['score'])-expected['score']),
            optimizer_state_step_equal=got['optimizer_state_step']==expected['optimizer_state_step'])
        item['score_close'] = item['score_abs_error'] <= atol+rtol*abs(expected['score'])
        for key in ['parameter_before','gradient','parameter','exp_avg','exp_avg_sq']:
            a,b=got[key],expected[key]
            item[key+'_max_abs_error']=float((a-b).abs().max().item())
            item[key+'_close']=torch.allclose(a,b,atol=atol,rtol=rtol)
        item['passed']=item['projection_ids_exact'] and item['optimizer_state_step_equal'] and item['score_close'] and all(item[k+'_close'] for k in ['parameter_before','gradient','parameter','exp_avg','exp_avg_sq'])
        checks.append(item)
    return dict(all_passed=all(r['passed'] for r in checks),steps=len(checks),
                source_commit=COMMIT,uses_unmodified_original_nn_project=True,
                allowed_pool_adaptation='same allowed IDs as the actual minimal program',
                atol=atol,rtol=rtol,checks=checks)


def check_native_score_gradient(clip, tokenizer, target_feature, token_ids, score_fn,
                                atol=1e-6, rtol=1e-5):
    """Independently check custom embedded-input scoring against native HF forward.

    A hook replaces only token lookup output, so native position embeddings,
    attention masking, EOS pooling and text projection remain unchanged.
    No model parameter is trained or modified.
    """
    table = clip.text_model.embeddings.token_embedding
    device = table.weight.device
    ids = torch.as_tensor(token_ids, device=device, dtype=torch.long).reshape(1,-1)
    input_ids = torch.cat([ids.new_tensor([[tokenizer.bos_token_id]]), ids,
                           ids.new_tensor([[tokenizer.eos_token_id]])],dim=1)
    embedded = table(ids).detach()
    manual_input = embedded.clone().requires_grad_(True)
    manual_score = score_fn(manual_input).mean()
    manual_gradient, = torch.autograd.grad(manual_score,[manual_input])
    native_input = embedded.clone().requires_grad_(True)
    bos = table(input_ids[:,:1]).detach()
    eos = table(input_ids[:,-1:]).detach()
    padded = torch.cat([bos,native_input,eos],dim=1)
    handle = table.register_forward_hook(lambda module,args,output:padded)
    try:
        features = clip.get_text_features(input_ids=input_ids)
    finally:
        handle.remove()
    features = features / features.norm(dim=-1,keepdim=True)
    target = target_feature / target_feature.norm(dim=-1,keepdim=True)
    native_score = (features @ target.T).mean()
    native_gradient, = torch.autograd.grad(native_score,[native_input])
    result=dict(score_max_abs_error=float((manual_score-native_score).abs().item()),
        gradient_max_abs_error=float((manual_gradient-native_gradient).abs().max().item()),
        score_close=torch.allclose(manual_score,native_score,atol=atol,rtol=rtol),
        gradient_close=torch.allclose(manual_gradient,native_gradient,atol=atol,rtol=rtol))
    result['all_passed']=result['score_close'] and result['gradient_close']
    return result
