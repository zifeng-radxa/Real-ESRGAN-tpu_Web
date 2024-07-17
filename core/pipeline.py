from core.imageupscaler import ImageUpscaler
import os
import uuid
from tools.utils import get_model_list
import argparse
import gradio as gr
model_list = get_model_list()
image_upscaler = ImageUpscaler()

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



def image_pipeline(input_path, model, face_enhance=None, background_remove=None, output_path=None, save=False):
    if save:
        if output_path is None:
            if not os.path.exists(os.path.join('./result', 'image')):
                os.makedirs(os.path.join('./result', 'image'), exist_ok=True)
            output_path = './result/image/out_{}.jpg'.format(uuid.uuid4())
    if input_path is None:
        return ("Please upload image", None)

    image_upscaler.change_model(model, face_enhance)
    res = image_upscaler(input_path, output_path, face_enhance)

    return ("Success upscale, click download icon to download to local", res)




# def video_pipeline(input_path, model, num_worker=1, face_enhance=None, background_remove=None):
#     if not os.path.exists(os.path.join('./result', 'video')):
#         os.makedirs(os.path.join('./result', 'video'), exist_ok=True)
#
#     output_path = './result/video/out_{}.mp4'.format(uuid.uuid4())
#     tmp_path = './result/video/temph264_{}.mp4'.format(uuid.uuid4())
#
#     up = VideoUpscale()



