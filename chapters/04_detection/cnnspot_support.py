"""CNNSpot 的权重校验、下载与证据记录；不分发第三方权重。"""
from pathlib import Path
import hashlib, json, os, sys, urllib.request
SHA256 = '3533f3e58a3b82215f21197f2faf4e99b911ec42a4a6e2d7823e04d5f3fd245a'
URL = 'https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1'
SOURCE = 'https://github.com/PeterWang512/CNNDetection'
REVISION = 'ea0b5622365e3a9cd31d1b54b6b5971131a839ab'
def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(f.read()).hexdigest()
def load_weights():
    path = Path(os.environ.get('CNN_SPOT_WEIGHTS', 'results/detection_weights/blur_jpg_prob0.5.pth'))
    if not path.is_file():
        raise FileNotFoundError('先运行 cnnspot_support.py --download，或设置 CNN_SPOT_WEIGHTS 指向官方权重。')
    if digest(path) != SHA256:
        raise ValueError('权重 SHA-256 不匹配，停止运行。')
    return path

def save_manifest(out, weights, fixtures, inputs, script):
    import torch, torchvision, PIL
    record = dict(method='CNNSpot, spatial CNN binary classification', architecture='ResNet-50, one logit',
        source=SOURCE, source_revision=REVISION, weight_sha256=digest(weights),
        code_sha256=digest(script), support_sha256=digest(__file__),
        python=sys.version, torch=torch.__version__, torchvision=torchvision.__version__, pillow=PIL.__version__,
        device='cpu', dtype='float32', threshold=.5, score='sigmoid(logit), higher=fake',
        preprocessing='RGB; ToTensor; ImageNet normalize; no resize/crop',
        inputs=[dict(file=n, source=s, sha256=digest(fixtures/n)) for n,s in inputs],
        boundary='Two source images; not an accuracy or generalization evaluation.')
    (out/'manifest.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf8')

if __name__ == '__main__':
    if sys.argv[1:] != ['--download']:
        raise SystemExit('用法：python cnnspot_support.py --download；先阅读 README 中的来源与许可。')
    target=Path(os.environ.get('CNN_SPOT_WEIGHTS', 'results/detection_weights/blur_jpg_prob0.5.pth'))
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temp=target.with_suffix('.download')
        urllib.request.urlretrieve(URL, temp)
        if digest(temp) != SHA256:
            temp.unlink()
            raise ValueError('下载权重校验失败。')
        temp.replace(target)
    print(load_weights())
