from core.imageupscaler import ImageUpscaler
import os
import uuid
from tools.utils import get_model_list, timer

import cv2
import numpy as np
from plugin.bgremove import Bgremover
model_list = get_model_list()
image_upscaler = ImageUpscaler()
bger = Bgremover()


# def run(input_path, model, type, num_worker=1, audio_check=None, face_enhance=None):
#     if not os.path.exists(os.path.join('./result', type)):
#         os.makedirs(os.path.join('./result', type), exist_ok=True)
#     if type == "video":
#         output_path = './result/video/out_{}.mp4'.format(uuid.uuid4())
#         tmp_path = './result/video/temph264_{}.mp4'.format(uuid.uuid4())
#     else:
#         output_path = './result/image/out_{}.jpg'.format(uuid.uuid4())
#         tmp_path = None
#     # try:
#     up = Upscale(input_path, output_path, model, tmp_path, type, num_worker=num_worker, face_enhance=face_enhance)
#     result_path = up(audio_check)
#     # except Exception as e:
#     #     print(e)
#     #     gr.Error("Error, please check the info box")
#     #     return (e, None)
#
#     return ("Success upscale, click download icon to download to local", result_path)


@timer
def image_pipeline(input, model, face_enhance=None, background_remove=None, output_path=None, save=False):
    if input is None:
        return ("Please upload image", None)

    if isinstance(input, str):
        img = cv2.imread(input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(input, np.ndarray):
        img = input

    if background_remove == 0:
        image_upscaler.change_model(model, face_enhance_name="None", bg_helper=True)
        bg_hlepr_img = image_upscaler(img, face_enhance="None", bg_upscale=True)
        bger.init_model()
        _, mask = bger.forward(bg_hlepr_img)
        image_upscaler.change_model(model, face_enhance_name=face_enhance, bg_helper=False)
        res = image_upscaler(img, face_enhance)
        new_mask = cv2.resize(mask, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_LINEAR)
        res = cv2.cvtColor(res, cv2.COLOR_RGB2BGRA)
        res = res[:,:,[2,1,0,3]]
        res[:,:,3] = new_mask

    elif background_remove == 1:
        # only bgremove
        bger.init_model()
        res, _ = bger.forward(img)

    elif background_remove == 2:
        # only upscale
        image_upscaler.change_model(model, face_enhance, bg_helper=False)
        res = image_upscaler(img, face_enhance, bg_upscale=False)

    if save:
        if output_path is None:
            if not os.path.exists(os.path.join('./result', 'image')):
                os.makedirs(os.path.join('./result', 'image'), exist_ok=True)
            output_path = './result/image/out_{}.jpg'.format(uuid.uuid4())

        if res.shape[2] == 4:
            cv2.imwrite(output_path, res[:,:,[2,1,0,3]])
        else:
            res = cv2.cvtColor(res, cv2.COLOR_RGBA2BGR)
            cv2.imwrite(output_path, res)

    return ("Success, click download icon to download", res)

# def video_pipeline(input_path, model, num_worker=1, face_enhance=None, background_remove=None):
#     if not os.path.exists(os.path.join('./result', 'video')):
#         os.makedirs(os.path.join('./result', 'video'), exist_ok=True)
#
#     output_path = './result/video/out_{}.mp4'.format(uuid.uuid4())
#     tmp_path = './result/video/temph264_{}.mp4'.format(uuid.uuid4())
#
#     up = VideoUpscale()



