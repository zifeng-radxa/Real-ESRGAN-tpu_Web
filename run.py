from upscale import Upscale
import os
import uuid

model_list = ['RealESRGAN_x4plus_rgb_1684x_BF16.bmodel',
              'RealESRGAN_x4plus_anime_6B_rgb_bm1684x_BF16.bmodel',
              'realesr-animevideov3_rgb_bm1684x_BF16.bmodel',
              'realesr-general-x4v3_dni_0_2_rgb_1684x_BF16.bmodel',
              ]

def run(input_path, model, type, num_worker=1, audio_check=None, face_enhance=None):
    if not os.path.exists(os.path.join('./result', type)):
        os.makedirs(os.path.join('./result', type), exist_ok=True)
    if type == "video":
        output_path = './result/video/out_{}.mp4'.format(uuid.uuid4())
        tmp_path = './result/video/temph264_{}.mp4'.format(uuid.uuid4())
    else:
        output_path = './result/image/out_{}.jpg'.format(uuid.uuid4())
        tmp_path = None
    # try:
    up = Upscale(input_path, output_path, model, tmp_path, type, num_worker=num_worker, face_enhance=face_enhance)
    result_path = up(audio_check)
    # except Exception as e:
    #     print(e)
    #     gr.Error("Error, please check the info box")
    #     return (e, None)

    return ("Success upscale, click download icon to download to local", result_path)

