from tools.utils import download_file
from tqdm import tqdm

models_url = {
    #"GFPGANv1.3.pth": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/GFPGANv1.3.pth",
    "parsing_parsenet_rgb_1_3_512_512.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/parsing_parsenet_rgb_1_3_512_512.bmodel",
    "realesr-animevideo_v3_rgb_1_3_480_640.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/realesr-animevideo_v3_rgb_1_3_480_640.bmodel",
    "realesr-general-x4v3_dni_0_2_rgb_1_3_480_640.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/realesr-general-x4v3_dni_0_2_rgb_1_3_480_640.bmodel",
    "RealESRGAN_x4plus_anime_6B_rgb_1_3_480_640.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/RealESRGAN_x4plus_anime_6B_rgb_1_3_480_640.bmodel",
    "RealESRGAN_x4plus_BF16_rgb_1_3_480_640.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/RealESRGAN_x4plus_BF16_rgb_1_3_480_640.bmodel",
    "retinaface_resnet50_rgb_1_3_480_640.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/retinaface_resnet50_rgb_1_3_480_640.bmodel",
    "codeformer_1-3-512-512_1-235ms.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/codeformer_1-3-512-512_1-235ms.bmodel"
}

def download_models():
    tqdm_tool = tqdm(total=len(models_url))
    for name, url in models_url.items():
        download_file(url,name)
        tqdm_tool.update(1)

download_models()