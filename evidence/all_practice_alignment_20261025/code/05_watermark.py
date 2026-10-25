from pathlib import Path
import os
import json
import cv2
import numpy as np

root = Path(os.environ['LAB_ROOT'])
out = root / 'minimal_examples_20261024/05_watermark'
out.mkdir(parents=True, exist_ok=True)
image = cv2.imread(str(root / 'minimal_examples_20261024/02_generation/red_2026.png'))
bits = np.random.default_rng(2026).integers(0, 2, 16)
coords = [(r, c) for r in range(0, 512, 8) for c in range(0, 512, 8)]
# 在亮度块的两个DCT系数差中循环嵌入16位消息，强度固定16。
ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype('float32')
for k, (r, c) in enumerate(coords):
    d = cv2.dct(ycc[r:r+8, c:c+8, 0])
    change = (2 * bits[k % 16] - 1) * 16 - (d[2, 3] - d[3, 2])
    d[2, 3] += change / 2
    d[3, 2] -= change / 2
    ycc[r:r+8, c:c+8, 0] = cv2.idct(d)
marked = cv2.cvtColor(np.clip(ycc, 0, 255).astype('uint8'), cv2.COLOR_YCrCb2BGR)
cv2.imwrite(str(out / 'marked.png'), marked)
cv2.imwrite(str(out / 'marked_q75.jpg'), marked, [cv2.IMWRITE_JPEG_QUALITY, 75])
# 分别解码PNG、JPEG和未嵌入原图；同一位在不同块中累积投票。
rows = []
for name, test in [('png', cv2.imread(str(out / 'marked.png'))),
                   ('jpeg75', cv2.imread(str(out / 'marked_q75.jpg'))), ('unmarked', image)]:
    lum = cv2.cvtColor(test, cv2.COLOR_BGR2YCrCb)[:, :, 0]
    votes = np.zeros(16)
    for k, (r, c) in enumerate(coords):
        d = cv2.dct(lum[r:r+8, c:c+8].astype('float32'))
        votes[k % 16] += d[2, 3] - d[3, 2]
    recovered = (votes > 0).astype(int)
    rows.append(dict(condition=name, source_bits=''.join(map(str, bits)),
                     decoded_bits=''.join(map(str, recovered)),
                     bit_accuracy=float(np.mean(bits == recovered))))
(out / 'decoded.json').write_text(json.dumps(rows, indent=2))
print(rows)
