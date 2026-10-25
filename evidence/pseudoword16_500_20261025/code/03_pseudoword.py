import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline

root = Path(os.environ['LAB_ROOT'])
models = json.loads((root / 'manifests/models.json').read_text())
out = root / 'pseudoword16_500'
out.mkdir(parents=True, exist_ok=True)
clip = CLIPModel.from_pretrained(models['clip']['path'], attn_implementation='eager',
                                 local_files_only=True).eval().to('cuda').requires_grad_(False)
processor = CLIPProcessor.from_pretrained(models['clip']['path'], local_files_only=True)
tok = processor.tokenizer
# 人工白名单只含中性音节，不含杯器、饮品、颜色或材质的明确描述词。
words = 'flo ri ak tor li vo sar vi lub ra daz mi fon za mel ku dax fim fino lem mim nom pim rem'.split()
vocab = torch.tensor([tok(w, add_special_tokens=False).input_ids[0] for w in words], device='cuda')
target = Image.open(root / 'inputs/pseudoword_target.png').convert('RGB')
target.save(out / 'target.png')
with torch.no_grad():
    pixels = processor(images=target, return_tensors='pt').pixel_values.to('cuda')
    image_feature = F.normalize(clip.get_image_features(pixels), dim=-1)
    table = clip.text_model.embeddings.token_embedding.weight
    basis = table[vocab]
    ends = table[torch.tensor([tok.bos_token_id, tok.eos_token_id], device='cuda')]
mask = torch.full((1, 1, 18, 18), float('-inf'), device='cuda').triu(1)

def text_feature(weights):
    # BOS + 16个可优化词元 + EOS；在EOS位置汇聚，与标准CLIP池化一致。
    embeds = torch.cat([ends[:1], weights @ basis, ends[1:]])[None]
    hidden = clip.text_model.embeddings(inputs_embeds=embeds)
    hidden = clip.text_model.encoder(inputs_embeds=hidden,
        causal_attention_mask=mask, return_dict=True).last_hidden_state
    pooled = clip.text_model.final_layer_norm(hidden)[:, 17]
    return F.normalize(clip.text_projection(pooled), dim=-1)

torch.manual_seed(2026)
logits = torch.nn.Parameter(torch.randn(16, len(words), device='cuda') * 0.01)
optimizer = torch.optim.Adam([logits], lr=0.1)
trace = []
for step in range(501):
    soft = logits.softmax(dim=-1)
    hard = F.one_hot(soft.argmax(dim=-1), len(words)).float()
    # 前向使用真实离散词元；反向用softmax梯度近似，称为直通估计。
    feature = text_feature(hard - soft.detach() + soft)
    score = (feature @ image_feature.T).squeeze()
    ids = vocab[hard.argmax(dim=-1)].tolist()
    trace.append(dict(step=step, score=score.item(), token_ids=ids, text=tok.decode(ids)))
    if step < 500:
        optimizer.zero_grad()
        (-score).backward()
        optimizer.step()
best = max(trace, key=lambda row: row['score'])
(out / 'search.json').write_text(json.dumps(dict(vocabulary=words, seed=2026,
    tokens=16, updates=500, learning_rate=0.1, trace=trace, best=best), indent=2))
print('initial/final/best:', trace[0], trace[-1], best, flush=True)

pipe = StableDiffusionPipeline.from_pretrained(models['sd15']['path'],
    torch_dtype=torch.float16, local_files_only=True).to('cuda')
rows = []
for name, prompt in [('initial', trace[0]['text']), ('best', best['text']),
                     ('natural', 'a red ceramic cup on a wooden table')]:
    result = pipe(prompt, generator=torch.Generator('cpu').manual_seed(2026),
                  num_inference_steps=30, guidance_scale=7.5, height=512, width=512)
    image = result.images[0]
    image.save(out / f'{name}.png')
    with torch.no_grad():
        pixels = processor(images=image, return_tensors='pt').pixel_values.to('cuda')
        generated = F.normalize(clip.get_image_features(pixels), dim=-1)
        text = processor(text=prompt, return_tensors='pt').to('cuda')
        encoded = F.normalize(clip.get_text_features(**text), dim=-1)
    rows.append(dict(name=name, prompt=prompt,
        token_ids=pipe.tokenizer(prompt, add_special_tokens=False).input_ids,
        text_target_cosine=(encoded @ image_feature.T).item(),
        image_target_cosine=(generated @ image_feature.T).item(),
        output_checker=result.nsfw_content_detected))
(out / 'comparison.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2), flush=True)
