"""独立观察三类过滤器；只做审核，不调用图像生成或攻击优化。"""
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import unicodedata


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def keyword_check(text, keywords):
    # 标准化大小写与空白；这只是可解释的教学词表。
    normalized = ' '.join(unicodedata.normalize('NFKC', text).casefold().split())
    hits = [word for word in keywords
            if re.search(r'(?<!\w)' + re.escape(word.casefold()) + r'(?!\w)', normalized)]
    return {'status': 'blocked' if hits else 'clear', 'matches': ';'.join(hits)}


def risk_result(scores, risk_label, threshold):
    # 查找指定风险类别的分数，不能把最高分类分数直接当作风险分数。
    labels = {item['label'].casefold(): float(item['score']) for item in scores}
    if risk_label.casefold() not in labels:
        raise ValueError('Risk label missing: ' + risk_label)
    value = labels[risk_label.casefold()]
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Invalid risk score')
    return {'status': 'blocked' if value >= threshold else 'clear',
            'score': value, 'threshold': threshold}


def validate_inputs(config, cases, images):
    if not config['keywords'] or any(not x.strip() for x in config['keywords']):
        raise ValueError('Keyword list must contain nonempty strings')
    for name in ('text_model', 'image_model'):
        item = config[name]
        if not 0 <= item['threshold'] <= 1 or not re.fullmatch(r'[0-9a-f]{40}', item['revision']):
            raise ValueError('Require a threshold in [0,1] and pinned revision')
    for rows in (cases, images):
        ids = [x['id'] for x in rows]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate sample IDs')
    if not cases or any(not x['prompt'].strip() for x in cases):
        raise ValueError('Prompt samples must be nonempty')


def load_classifier(kind, config, device):
    # 延迟导入：词表模式不需要安装或下载模型。
    from transformers import pipeline
    spec = config[kind + '_model']
    local = os.environ.get('FILTER_' + kind.upper() + '_MODEL_PATH')
    args = {'model': local or spec['id'], 'device': device}
    if not local:
        args['revision'] = spec['revision']
    task = 'text-classification' if kind == 'text' else 'image-classification'
    return pipeline(task, **args)


def save_csv(path, rows, fields):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def compare(folder, out, keyword_only=False, image_manifest=None, device=-1,
            loader=load_classifier):
    folder, out = Path(folder), Path(out)
    config = read_json(folder / 'filter_config.json')
    cases = read_json(folder / 'filter_cases.json')
    manifest = Path(image_manifest).resolve() if image_manifest else folder / 'filter_images.json'
    images = read_json(manifest)
    validate_inputs(config, cases, images)
    out.mkdir(parents=True, exist_ok=True)
    classifiers, errors = {}, {}
    for kind in ('text', 'image'):
        if not keyword_only:
            try:
                classifiers[kind] = loader(kind, config, device)
            except Exception as exc:
                errors[kind] = type(exc).__name__ + ': ' + str(exc)
    text_rows, image_rows, raw = [], [], {'text': {}, 'image': {}}
    failures = len(errors)
    for item in cases:
        key = keyword_check(item['prompt'], config['keywords'])
        row = {'id': item['id'], 'group': item['group'], 'prompt': item['prompt'],
               'keyword_status': key['status'], 'keyword_matches': key['matches'],
               'text_status': 'not_run' if keyword_only else 'unavailable',
               'nsfw_score': '', 'threshold': config['text_model']['threshold'],
               'error': errors.get('text', '')}
        if 'text' in classifiers:
            try:
                scores = classifiers['text'](item['prompt'], top_k=None, truncation=False)
                result = risk_result(scores, config['text_model']['risk_label'], row['threshold'])
                raw['text'][item['id']] = scores
                row.update(text_status=result['status'], nsfw_score=result['score'])
            except Exception as exc:
                failures += 1
                row['error'] = type(exc).__name__ + ': ' + str(exc)
        text_rows.append(row)
    for item in images:
        row = {'id': item['id'], 'group': item['group'], 'path': item['path'],
               'source': item['source'], 'sha256': '',
               'image_status': 'not_run' if keyword_only else 'unavailable',
               'nsfw_score': '', 'threshold': config['image_model']['threshold'],
               'error': errors.get('image', '')}
        if 'image' in classifiers:
            try:
                from PIL import Image
                path = Path(item['path'])
                if not path.is_absolute():
                    path = manifest.parent / path
                row['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
                with Image.open(path) as original:
                    scores = classifiers['image'](original.convert('RGB'), top_k=None)
                result = risk_result(scores, config['image_model']['risk_label'], row['threshold'])
                raw['image'][item['id']] = scores
                row.update(image_status=result['status'], nsfw_score=result['score'])
            except Exception as exc:
                failures += 1
                row['error'] = type(exc).__name__ + ': ' + str(exc)
        image_rows.append(row)
    save_csv(out / 'text_results.csv', text_rows, list(text_rows[0]))
    save_csv(out / 'image_results.csv', image_rows,
             ['id','group','path','source','sha256','image_status','nsfw_score','threshold','error'])
    summary = {'text_samples': len(text_rows), 'image_samples': len(image_rows),
               'keyword_blocked': sum(x['keyword_status'] == 'blocked' for x in text_rows),
               'text_blocked': sum(x['text_status'] == 'blocked' for x in text_rows),
               'image_blocked': sum(x['image_status'] == 'blocked' for x in image_rows),
               'text_scored': sum(x['text_status'] in ('blocked','clear') for x in text_rows),
               'image_scored': sum(x['image_status'] in ('blocked','clear') for x in image_rows),
               'failures': failures, 'keyword_only': keyword_only,
               'image_positive_control_present': any(x['group'] == 'risk' for x in image_rows),
               'boundary': 'Independent audit; group labels are teaching annotations. Clear does not prove safety.'}
    versions = {}
    for name in ('torch','transformers','Pillow','numpy'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    for name, value in [('summary.json', summary), ('raw_scores.json', raw),
                        ('settings.json', {'config': config, 'device': device, 'versions': versions,
                         'python': sys.version,
                         'loaded_versions': {name: getattr(sys.modules.get(name), '__version__', None)
                           for name in ('torch','transformers','PIL','numpy')},
                         'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'local_model_overrides': {k: os.environ.get('FILTER_'+k.upper()+'_MODEL_PATH')
                                                   for k in ('text','image')},
                         'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [folder/'filter_config.json',folder/'filter_cases.json',manifest]}})]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary
