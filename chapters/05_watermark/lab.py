from pathlib import Path; import csv; import cv2
import numpy as np
out = Path('results/watermark'); out.mkdir(parents=True, exist_ok=True)
# 1. 优先读取生成图；缺失时改用程序合成测试图。
path = Path('results/generation/cup_0.png')
if not path.exists():
    y, x = np.mgrid[:512, :512]
    image = np.stack([(x+y)%256, x%256, y%256], axis=2).astype('uint8')
else:
    image = cv2.imread(str(path))
image = cv2.resize(image, (512, 512)); cv2.imwrite(str(out/'input.png'), image)
# 2. 转到亮度通道，在每个 8×8 块中嵌入水印。
ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb).astype('float32')
# 固定种子生成 16 位消息；强度控制系数差的幅度。
bits = np.random.default_rng(2026).integers(0, 2, 16); strength = 24
for k, (r, c) in enumerate((r,c) for r in range(0,512,8) for c in range(0,512,8)):
    block = cv2.dct(ycc[r:r+8,c:c+8,0]); a, b = block[2,3], block[3,2]
    # 调整两个 DCT 系数，使差值正负表示消息位。
    delta = (2*bits[k%16]-1)*strength - (a-b)
    block[2,3] += delta/2; block[3,2] -= delta/2
    ycc[r:r+8,c:c+8,0] = cv2.idct(block)
watermarked = cv2.cvtColor(np.clip(ycc,0,255).astype('uint8'), cv2.COLOR_YCrCb2BGR)
cv2.imwrite(str(out/'marked.png'), watermarked); rows = []
# 3. 对带水印图重新压缩，检验消息能否恢复。
for quality in (100, 75, 40):
    cv2.imwrite(str(out/f'q{quality}.jpg'),watermarked,[cv2.IMWRITE_JPEG_QUALITY,quality])
    test = cv2.imread(str(out/f'q{quality}.jpg')); lum=cv2.cvtColor(test,cv2.COLOR_BGR2YCrCb)[:,:,0]
    # 对重复嵌入的同一位累加系数差，再按正负判决。
    votes = np.zeros(16)
    for k,(r,c) in enumerate((r,c) for r in range(0,512,8) for c in range(0,512,8)):
        d=cv2.dct(lum[r:r+8,c:c+8].astype('float32')); votes[k%16] += d[2,3]-d[3,2]
    # 记录位恢复率与 PSNR；后者同时反映水印和压缩影响。
    rows.append((quality,float(np.mean((votes>0)==bits)),float(cv2.PSNR(image,test))))
# 4. 保存结果；单张图的恢复率不能代表通用性能。
with (out/'scores.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerows([['jpeg_quality','bit_accuracy','psnr_db'], *rows])
