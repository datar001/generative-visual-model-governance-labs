"""无害红杯目标的伪词搜索类比；仅计算CLIP匹配，不调用真实安全过滤或生成器。"""
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def classroom_gate(prompt, excluded):
    # red/cup没有风险含义，只用来观察字面规则与图文表示的差别。
    words = prompt.casefold().split()
    return not any(word.casefold() in words for word in excluded)


def search(syllables, length, seed, budget, mode, evaluate):
    if mode not in ('random', 'improve'):
        raise ValueError('Unknown search mode')
    candidates = list(itertools.product(syllables, repeat=length))
    if not 1 <= budget <= len(candidates):
        raise ValueError('Budget must fit the finite candidate space')
    rng = random.Random(seed)
    rng.shuffle(candidates)
    initial = candidates[0]
    current, best_score = initial, -math.inf
    seen, trace = set(), []
    for query in range(1, budget + 1):
        if query == 1:
            candidate = initial
        elif mode == 'random':
            candidate = next(x for x in candidates if x not in seen)
        else:
            # 仅在固定无害教学词表里替换一个位置，观察目标分数是否提高。
            neighbors = [current[:i] + (word,) + current[i+1:]
                         for i in range(length) for word in syllables
                         if word != current[i] and current[:i] + (word,) + current[i+1:] not in seen]
            candidate = rng.choice(neighbors) if neighbors else next(x for x in candidates if x not in seen)
        seen.add(candidate)
        prompt = ' '.join(candidate)
        value = float(evaluate(prompt))
        if not math.isfinite(value):
            raise ValueError('Scoring failed; cannot rank a nonfinite value')
        if value > best_score:
            current, best_score = candidate, value
        trace.append({'seed':seed, 'budget':budget, 'mode':mode, 'query':query,
                      'candidate':prompt, 'score':value,
                      'best_candidate':' '.join(current), 'best_score':best_score})
    return trace


class CupScorer:
    def __init__(self, config, folder):
        # 编码器固定在CPU上，不训练模型，也不调用LLM API。
        import torch
        from transformers import CLIPModel, CLIPProcessor
        from PIL import Image
        self.torch, self.cache = torch, {}
        source = os.environ.get('PSEUDOWORD_CLIP_PATH', config['model'])
        kwargs = {} if 'PSEUDOWORD_CLIP_PATH' in os.environ else {'revision':config['revision']}
        self.model = CLIPModel.from_pretrained(source, **kwargs).eval()
        self.processor = CLIPProcessor.from_pretrained(source, **kwargs)
        with Image.open(folder / config['target']) as image:
            inputs = self.processor(images=image.convert('RGB'), return_tensors='pt')
        with torch.inference_mode():
            feature = self.model.get_image_features(**inputs)
            self.target = feature / feature.norm(dim=-1, keepdim=True)

    def __call__(self, prompt):
        if prompt not in self.cache:
            inputs = self.processor(text=[prompt], padding=True, return_tensors='pt')
            if inputs['input_ids'].shape[-1] > 77:
                raise ValueError('Text exceeds CLIP token limit')
            with self.torch.inference_mode():
                feature = self.model.get_text_features(**inputs)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                self.cache[prompt] = float((feature @ self.target.T).item())
        return self.cache[prompt]


def run(folder, out):
    folder, out = Path(folder), Path(out)
    config = json.loads((folder/'pseudoword_config.json').read_text(encoding='utf-8'))
    target = folder / config['target']
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    scorer = CupScorer(config, folder)
    natural = scorer(config['target_description'])
    rows, summary = [], []
    for seed in config['seeds']:
        for budget in config['budgets']:
            for mode in ('random', 'improve'):
                trace = search(config['syllables'], config['length'], seed, budget, mode, scorer)
                for row in trace:
                    row['classroom_gate'] = classroom_gate(row['candidate'], config['classroom_excluded_words'])
                rows.extend(trace)
                last = trace[-1]
                summary.append({'seed':seed, 'budget':budget, 'mode':mode,
                                'initial_score':trace[0]['score'], 'best_score':last['best_score'],
                                'best_candidate':last['best_candidate'], 'queries':len(trace)})
    with (out/'trace.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with (out/'summary.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    result = {'runs':len(summary), 'logical_queries':len(rows),
              'unique_text_encodings':len(scorer.cache),
              'natural_description_score':natural,
              'natural_description_passes_classroom_gate':classroom_gate(config['target_description'], config['classroom_excluded_words']),
              'all_candidates_pass_classroom_gate':all(r['classroom_gate'] for r in rows),
              'seconds':time.perf_counter()-started,
              'scope':config['scope'], 'real_safety_filter_tested':False,
              'text_to_image_generation_tested':False, 'llm_api_calls':0}
    settings = {'config':config, 'target_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                'implementation_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'python':sys.version, 'loaded_versions':{n:getattr(sys.modules.get(n),'__version__',None)
                                                     for n in ('torch','transformers','PIL','numpy')}}
    for name, data in [('result.json',result),('settings.json',settings)]:
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    return result
