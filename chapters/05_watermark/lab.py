from pathlib import Path; import csv; import cv2
import numpy as np
out = Path('results/watermark'); out.mkdir(parents=True, exist_ok=True)
path = Path('results/generation/cup_0.png')
if not path.exists():
    y, x = np.mgrid[:512, :512]
    image = np.stack([(x+y)%256, x%256, y%256], axis=2).astype('uint8')
else:
    image = cv2.imread(str(path))
image = cv2.resize(image, (512, 512)); cv2.imwrite(str(out/'input.png'), image)
ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype('float32')
bits = np.random.default_rng(2026).integers(0, 2, 16); strength = 24
for k, (r, c) in enumerate((r,c) for r in range(0,512,8) for c in range(0,512,8)):
    block = cv2.dct(ycc[r:r+8,c:c+8,0]); a, b = block[2,3], block[3,2]
    delta = (2*bits[k%16]-1)*strength - (a-b)
    block[2,3] += delta/2; block[3,2] -= delta/2
    ycc[r:r+8,c:c+8,0] = cv2.idct(block)
watermarked = cv2.cvtColor(np.clip(ycc,0,255).astype('uint8'), cv2.COLOR_YCrCb2BGR)
cv2.imwrite(str(out/'marked.png'), watermarked); rows = []
for quality in (100, 75, 40):
    cv2.imwrite(str(out/f'q{quality}.jpg'),watermarked,[cv2.IMWRITE_JPEG_QUALITY,quality])
    test = cv2.imread(str(out/f'q{quality}.jpg')); lum=cv2.cvtColor(test,cv2.COLOR_BGR2YCrCb)[:,:,0]
    votes = np.zeros(16)
    for k,(r,c) in enumerate((r,c) for r in range(0,512,8) for c in range(0,512,8)):
        d=cv2.dct(lum[r:r+8,c:c+8].astype('float32')); votes[k%16] += d[2,3]-d[3,2]
    rows.append((quality,float(np.mean((votes>0)==bits)),float(cv2.PSNR(image,test))))
with (out/'scores.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerows([['jpeg_quality','bit_accuracy','psnr_db'], *rows])
