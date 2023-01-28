import os
import cv2
import numpy as np
from PIL import Image

def create_rainbowgif(
    src_filename = "target.png",  # src_image
    mask_filename = None,  # mask of the src_image 
    output_filename = "target.png", 
    max_px = 1000,  # maximum pixel size of the output image
    time_frequency = 0.3, 
    spatial_frequency_x = 0.5, 
    spatial_frequency_y = 0.5, 
    duration = 3.0,
    fps = 30,
):

    # mask
    mask = None
    if mask_filename is not None and os.path.isfile(mask_filename):
        mask = cv2.imread(mask_filename, 0)

    # src_imgの読み込み　無理やり4chにする
    if not os.path.isfile(src_filename):
        print(f"{src_filename} is not found...")
        return False, None
    src = cv2.imread(src_filename, cv2.IMREAD_UNCHANGED)    
    if len(src.shape) == 2:  # srcが1chの場合
        src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    elif src.shape[2] == 4:  # srcが4chの場合
        mask = src[:, :, 3]
        src = src[:, :, :3]
    else:  # srcが3chの場合はそのまま
        pass

    # リサイズ
    Hi, Wi = src.shape[:2]  # 元画像サイズ
    gain = 1.0 if Hi <= max_px and Wi <= max_px else \
            max_px / Wi if Wi > Hi else \
            max_px / Hi
    Ho, Wo = ((int)(Hi * gain), (int)(Wi * gain))  # 保存時の画像サイズ
    src = cv2.resize(src, (Wo, Ho))
    if mask is None:
        mask = np.ones((Ho, Wo), np.uint8) * 255 
    else:
        mask = cv2.resize(mask, (Wo, Ho))
    mask = Image.fromarray(mask)
    mask = Image.eval(mask, lambda a: 255 if a <= 5 else 0)
    
    # HSV変換
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 前準備
    imgs = []
    total_frame = (int)(duration * fps + 0.5)
    # func(ft - kx - ky)
    kx = spatial_frequency_x   # x方向の空間周波数
    ky = spatial_frequency_y   # y方向の空間周波数
    f = time_frequency         # 時間周波数
    x = np.tile(np.arange(0, 1, 1 / Wo, np.float32), (Ho, 1))
    y = np.tile(np.arange(0, 1, 1 / Ho, np.float32), (Wo, 1)).T
    kxx_kyy = kx * x + ky * y

    for i in range(total_frame):
        t = i / fps  # time

        # linear-wave : A * func(ft - kx*x + ky*y) 0 ~ 1に正規化
        ft = np.ones((Ho, Wo), np.float32) * f * t
        theta = (ft - kxx_kyy) % 1  # 0 ~ 1
        h2 = (179 * theta).astype(np.uint8)   # hueのrangeは0~179

        h2sv = cv2.merge([h2, s, v])
        bgr = cv2.cvtColor(h2sv, cv2.COLOR_HSV2BGR)
        # alpha channelを事前に入れておいた方が処理が格段に速くなるのでalphaを入れる
        rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA)  # aはここでは255でよい
        dst = Image.fromarray(rgba)   # pillowに変換
        dst = dst.quantize(colors=256)   # gif用に色を削減
        dst.paste(im=255, mask=mask)     # 256色の1色を透過用に塗りつぶし
        imgs.append(dst)

    imgs[0].save(output_filename, save_all = True, append_images = imgs[1:], optimize = False, duration = 1000 // fps, loop = 0, transparency = 255)  # disposal = 2


if __name__ == "__main__":

    create_rainbowgif(
        src_filename = "demo/input.png", 
        mask_filename = None, #"mask.png", 
        output_filename = "demo/output.gif", 
        max_px = 1000,
        time_frequency = 2, 
        spatial_frequency_x = 1, 
        spatial_frequency_y = -1, 
        duration = 3.0,
        fps = 30
    )
