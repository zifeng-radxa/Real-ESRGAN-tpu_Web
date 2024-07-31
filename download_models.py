from tools.utils import load_file_from_url


models_url = {
    "video": {
        "rmbg_f16_video.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/rmbg_f16_video.bmodel",
        "realesr-general-x4v3_f16_video.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/realesr-general-x4v3_f16_video.bmodel",
        "realesr-animevideov3_f16_video.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/realesr-animevideov3_f16_video.bmodel"
    },

    "image": {
        "parsing_parsenet_rgb_1_3_512_512.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/parsing_parsenet_rgb_1_3_512_512.bmodel",
        "realesr-animevideo_v3_rgb_1_3_480_640.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/realesr-animevideo_v3_rgb_1_3_480_640.bmodel",
        "realesr-general-x4v3_dni_0_2_rgb_1_3_480_640.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/realesr-general-x4v3_dni_0_2_rgb_1_3_480_640.bmodel",
        "RealESRGAN_x4plus_anime_6B_rgb_1_3_480_640.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/RealESRGAN_x4plus_anime_6B_rgb_1_3_480_640.bmodel",
        "RealESRGAN_x4plus_BF16_rgb_1_3_480_640.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/RealESRGAN_x4plus_BF16_rgb_1_3_480_640.bmodel",
        "retinaface_resnet50_rgb_1_3_480_640.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/retinaface_resnet50_rgb_1_3_480_640.bmodel",
        "codeformer_1-3-512-512_1-235ms.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/codeformer_1-3-512-512_1-235ms.bmodel",
        "gfpgan.bmodel":  "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/gfpgan.bmodel",
        "rmbg_f16_1.bmodel": "https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web/releases/download/v0.1/rmbg_f16_1.bmodel"
    }

}

def download_models():
    for key, value in models_url["image"].items():
        load_file_from_url(value, "./models/image")

    for key, value in models_url["video"].items():
        load_file_from_url(value, "./models/video")

download_models()