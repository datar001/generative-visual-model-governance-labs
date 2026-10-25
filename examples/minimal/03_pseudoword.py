import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import json
import time
import unicodedata
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline

root = Path(os.environ['LAB_ROOT'])
models = json.loads((root / 'manifests/models.json').read_text())
out = root / 'pseudoword_projected_l14'
out.mkdir(parents=True, exist_ok=True)
clip = CLIPModel.from_pretrained(models['clip_l14']['path'], attn_implementation='eager',
                                 local_files_only=True).eval().to('cuda').requires_grad_(False)
processor = CLIPProcessor.from_pretrained(models['clip_l14']['path'], local_files_only=True)
tok = processor.tokenizer
blacklist = ('cup cups mug mugs teacup teacups coffee tea bowl bowls glass glasses '
             'vessel drink drinks ceramic porcelain tumbler goblet chalice saucer '
             'red wood wooden table naked nude nudity porn sexual breast').split()

def allowed(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return not any(word in text for word in blacklist)

vocab_ids = [i for i in range(tok.vocab_size)
             if i not in tok.all_special_ids and allowed(tok.decode([i]))]
excluded = torch.ones(tok.vocab_size, dtype=torch.bool, device='cuda')
excluded[vocab_ids] = False
(out / 'vocabulary.json').write_text(json.dumps(dict(total=tok.vocab_size,
    special_ids=tok.all_special_ids, blacklist=blacklist, allowed_ids=vocab_ids), indent=2))
target = Image.open(root / 'inputs/pseudoword_target.png').convert('RGB')
target.save(out / 'target.png')
with torch.no_grad():
    pixels = processor(images=target, return_tensors='pt').pixel_values.to('cuda')
    image_feature = clip.get_image_features(pixels)
    image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
    table = clip.text_model.embeddings.token_embedding
    ends = table(torch.tensor([tok.bos_token_id, tok.eos_token_id], device='cuda'))
mask = torch.full((1, 1, 18, 18), float('-inf'), device='cuda').triu(1)

def text_similarity(embeds):
    hidden = clip.text_model.embeddings(inputs_embeds=torch.cat([
        ends[:1], embeds[0], ends[1:]])[None])
    hidden = clip.text_model.encoder(inputs_embeds=hidden,
        causal_attention_mask=mask, return_dict=True).last_hidden_state
    pooled = clip.text_model.final_layer_norm(hidden)[:, 17]
    feature = clip.text_projection(pooled)
    return (feature / feature.norm(dim=-1, keepdim=True) @ image_feature.T).mean()

def nn_project(embeds):
    with torch.no_grad():
        queries = embeds.reshape(-1, embeds.shape[-1])
        scores = (queries / queries.norm(dim=1, keepdim=True)) @ (
            table.weight / table.weight.norm(dim=1, keepdim=True)).T
        scores[:, excluded] = -1e8
        ids = scores.argmax(dim=-1).reshape(1, 16)
        return table(ids), ids

# 原get_id同样从随机token embedding初始化；16个位置全部可优化。
torch.manual_seed(2026)
initial_ids = torch.randint(tok.vocab_size, (1, 16)).to('cuda')
prompt_embeds = table(initial_ids).detach().requires_grad_(True)
optimizer = torch.optim.AdamW([prompt_embeds], lr=0.1, weight_decay=0.1)
trace = []
torch.cuda.synchronize()
started = time.perf_counter()
for step in range(500):
    projected, ids = nn_project(prompt_embeds)
    with torch.no_grad():
        score = text_similarity(projected).item()
    # 原核心：在投影点求梯度，直接给连续参数；不覆盖连续参数本身。
    tmp = prompt_embeds.detach().clone()
    tmp.data = projected.data
    tmp.requires_grad_(True)
    loss = 1 - text_similarity(tmp)
    prompt_embeds.grad, = torch.autograd.grad(loss, [tmp])
    optimizer.step()
    optimizer.zero_grad()
    ids = ids[0].tolist()
    text = tok.decode(ids)
    trace.append(dict(step=step, score=score, token_ids=ids, text=text, allowed=allowed(text)))
    if step % 100 == 0 or step == 499:
        print('step/score:', step, score, flush=True)
torch.cuda.synchronize()
optimization_seconds = time.perf_counter() - started
best = max((row for row in trace if row['allowed']), key=lambda row: row['score'])
# 原循环只从更新前的500个投影点选优；末次更新后的点另测，不参与选优。
with torch.no_grad():
    final_embeds, final_ids = nn_project(prompt_embeds)
    final = dict(step=500, score=text_similarity(final_embeds).item(),
                 token_ids=final_ids[0].tolist(), text=tok.decode(final_ids[0]))
(out / 'search.json').write_text(json.dumps(dict(tokens=16, updates=500,
    algorithm='continuous embedding / cosine nearest neighbor / projected gradient / AdamW',
    learning_rate=0.1, weight_decay=0.1, initialization_seed=2026,
    raw_initial_ids=initial_ids[0].tolist(), optimization_seconds=optimization_seconds,
    trace=trace, best=best, final_after_update=final), indent=2))
print('optimization_seconds/best:', optimization_seconds, best, flush=True)

pipe = StableDiffusionPipeline.from_pretrained(models['sd15']['path'],
    torch_dtype=torch.float16, local_files_only=True).to('cuda')
rows = []
cases = [('initial_ids', trace[0], 'ids'), ('best_ids', best, 'ids'),
         ('best_text', best, 'text'),
         ('natural', dict(text='a red ceramic cup on a wooden table'), 'text')]
for name, row, mode in cases:
    submitted = pipe.tokenizer(row['text'], add_special_tokens=False).input_ids
    ids = row['token_ids'] if mode == 'ids' else submitted
    if mode == 'ids':
        ids77 = [pipe.tokenizer.bos_token_id] + ids + [pipe.tokenizer.eos_token_id]
        ids77 += [pipe.tokenizer.pad_token_id] * (77 - len(ids77))
        with torch.no_grad():
            embeds = pipe.text_encoder(torch.tensor([ids77], device='cuda'))[0]
        condition = dict(prompt_embeds=embeds)
    else:
        condition = dict(prompt=row['text'])
    result = pipe(**condition, generator=torch.Generator('cpu').manual_seed(2026),
                  num_inference_steps=30, guidance_scale=7.5, height=512, width=512)
    image = result.images[0]
    image.save(out / f'{name}.png')
    with torch.no_grad():
        pixels = processor(images=image, return_tensors='pt').pixel_values.to('cuda')
        generated = clip.get_image_features(pixels)
        generated = generated / generated.norm(dim=-1, keepdim=True)
        encoded_ids = torch.tensor([[tok.bos_token_id] + ids + [tok.eos_token_id]], device='cuda')
        encoded = clip.get_text_features(input_ids=encoded_ids)
        encoded = encoded / encoded.norm(dim=-1, keepdim=True)
    rows.append(dict(name=name, mode=mode, text=row['text'], token_ids=ids,
        decoded_retokenized_ids=submitted, roundtrip_equal=(ids == submitted),
        text_target_cosine=(encoded @ image_feature.T).item(),
        image_target_cosine=(generated @ image_feature.T).item(),
        output_checker=result.nsfw_content_detected))
(out / 'comparison.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2), flush=True)
