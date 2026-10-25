import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from pathlib import Path
import json
import unicodedata
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from diffusers import StableDiffusionPipeline

root = Path(os.environ['LAB_ROOT'])
models = json.loads((root / 'manifests/models.json').read_text())
out = root / 'pseudoword_l14'
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

# 完整词表仅排除特殊ID和黑名单命中项，不筛掉BPE碎片或非英语词元。
vocab_ids = [i for i in range(tok.vocab_size)
             if i not in tok.all_special_ids and allowed(tok.decode([i]))]
vocab = torch.tensor(vocab_ids, device='cuda')
(out / 'vocabulary.json').write_text(json.dumps(dict(total=tok.vocab_size,
    special_ids=tok.all_special_ids, blacklist=blacklist, allowed_ids=vocab_ids), indent=2))
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
    embeds = torch.cat([ends[:1], weights @ basis, ends[1:]])[None]
    hidden = clip.text_model.embeddings(inputs_embeds=embeds)
    hidden = clip.text_model.encoder(inputs_embeds=hidden,
        causal_attention_mask=mask, return_dict=True).last_hidden_state
    pooled = clip.text_model.final_layer_norm(hidden)[:, 17]
    return F.normalize(clip.text_projection(pooled), dim=-1)

# 初始离散提示沿用24词元实验；其余分数为0，选中项为0.1。
initial_text = 'ku li mim ri mel dax dax ku sar fon daz fon ri nom mel lem'
initial = tok(initial_text, add_special_tokens=False).input_ids
logits = torch.nn.Parameter(torch.zeros(16, len(vocab_ids), device='cuda'))
with torch.no_grad():
    logits[torch.arange(16), [vocab_ids.index(i) for i in initial]] = 0.1
optimizer = torch.optim.Adam([logits], lr=0.1)
trace = []
for step in range(501):
    soft = logits.softmax(dim=-1)
    hard = F.one_hot(soft.argmax(dim=-1), len(vocab_ids)).float()
    feature = text_feature(hard - soft.detach() + soft)
    score = (feature @ image_feature.T).squeeze()
    ids = vocab[hard.argmax(dim=-1)].tolist()
    text = tok.decode(ids)
    trace.append(dict(step=step, score=score.item(), token_ids=ids,
                      text=text, allowed=allowed(text)))
    if step < 500:
        optimizer.zero_grad()
        (-score).backward()
        optimizer.step()
best = max((row for row in trace if row['allowed']), key=lambda row: row['score'])
(out / 'search.json').write_text(json.dumps(dict(tokens=16, updates=500,
    learning_rate=0.1, initialization='same discrete IDs as 24-token experiment',
    trace=trace, best=best), indent=2))
print('initial/final/best:', trace[0], trace[-1], best, flush=True)

pipe = StableDiffusionPipeline.from_pretrained(models['sd15']['path'],
    torch_dtype=torch.float16, local_files_only=True).to('cuda')
rows = []
cases = [('initial_ids', trace[0], 'ids'), ('best_ids', best, 'ids'),
         ('best_text', best, 'text'),
         ('natural', dict(text='a red ceramic cup on a wooden table'), 'text')]
for name, row, mode in cases:
    text = row['text']
    submitted = pipe.tokenizer(text, add_special_tokens=False).input_ids
    ids = row['token_ids'] if mode == 'ids' else submitted
    if mode == 'ids':
        # CLIP投影与SD条件均为768维，但条件须用SD自身的逐位置输出。
        ids77 = [pipe.tokenizer.bos_token_id] + ids + [pipe.tokenizer.eos_token_id]
        ids77 += [pipe.tokenizer.pad_token_id] * (77 - len(ids77))
        with torch.no_grad():
            embeds = pipe.text_encoder(torch.tensor([ids77], device='cuda'))[0]
        condition = dict(prompt_embeds=embeds)
    else:
        condition = dict(prompt=text)
    result = pipe(**condition, generator=torch.Generator('cpu').manual_seed(2026),
                  num_inference_steps=30, guidance_scale=7.5, height=512, width=512)
    image = result.images[0]
    image.save(out / f'{name}.png')
    with torch.no_grad():
        pixels = processor(images=image, return_tensors='pt').pixel_values.to('cuda')
        generated = F.normalize(clip.get_image_features(pixels), dim=-1)
        encoded_ids = torch.tensor([[tok.bos_token_id] + ids + [tok.eos_token_id]], device='cuda')
        encoded = F.normalize(clip.get_text_features(input_ids=encoded_ids), dim=-1)
    rows.append(dict(name=name, mode=mode, text=text, token_ids=ids,
        decoded_retokenized_ids=submitted, roundtrip_equal=(ids == submitted),
        text_target_cosine=(encoded @ image_feature.T).item(),
        image_target_cosine=(generated @ image_feature.T).item(),
        output_checker=result.nsfw_content_detected))
(out / 'comparison.json').write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2), flush=True)
