import os
import uuid
import cv2
import numpy as np

from tools.utils import get_model_list, timer, fuse_audio_with_ffmpeg
from tools.tpu_utils import clean_tpu_memory
from core.imageupscaler import ImageUpscaler, ImageUpscaler2
from plugin.bgremove import Bgremover, Bgremover2

image_model_list = get_model_list()
video_model_list = get_model_list("video")
image_upscaler = ImageUpscaler()
image_upscaler2 = ImageUpscaler2()
bger = Bgremover()
bger2 = Bgremover2()

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
def image_pipeline(input, model, face_enhance=None, background_remove=None, output_path=None, save=False, thread=None):
    clean_tpu_memory([bger2])
    if input is None:
        return ("Please upload image", None)

    if isinstance(input, str):
        img = cv2.imread(input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(input, np.ndarray):
        img = input
    if background_remove == 0:
        image_upscaler.change_model(model, face_enhance_name="None", bg_helper=True)
        bg_hlepr_img = image_upscaler.forward(img, face_enhance="None", bg_upscale=True)
        bger.init_model()
        _, mask = bger.forward(bg_hlepr_img)
        image_upscaler.change_model(model, face_enhance_name=face_enhance, bg_helper=False)
        res = image_upscaler.forward(img, face_enhance)
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
        res = image_upscaler.forward(img, face_enhance, bg_upscale=False)

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

@timer
def video_pipeline(input, model, face_enhance=None, background_remove=None, thread=1, audio=None, output_path=None, save=True):
    clean_tpu_memory([image_upscaler, bger])
    if not isinstance(input, str) or input is None:
        print("Please upload your video")
        return "Please upload your video", None

    if input.endswith('.flv'):
        mp4_path = input.replace('.flv', '.mp4')
        os.system(f'ffmpeg_ubuntu -i {input} -codec copy {mp4_path}')
        input = mp4_path

    if not os.path.exists('./temp_frames'):
        os.makedirs('./temp_frames', exist_ok=True)
    if not os.path.exists('./temp_res_frames'):
        os.makedirs('./temp_res_frames', exist_ok=True)

    cap = cv2.VideoCapture(input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print("**************")
    os.system(f'ffmpeg_ubuntu -i {input} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  ./temp_frames/frame%d.png')
    print("**************")

    if background_remove == 0:
        pass

    elif background_remove == 1:
        bger2.init_model(thread_num= thread)
        bger2.forward('./temp_frames')

        pass

    elif background_remove == 2:
        # only upscale
        image_upscaler2.change_model(model, face_enhance, thread_num=thread)
        image_upscaler2.forward('./temp_frames')


    if save:
        if output_path is None:
            if not os.path.exists(os.path.join('./result', 'video')):
                os.makedirs(os.path.join('./result', 'video'), exist_ok=True)
            output_path = './result/video/out_{}.mp4'.format(uuid.uuid4())
        print("merge")
        os.system(f'ffmpeg_ubuntu -framerate {fps} -i ./temp_res_frames/frame%d.png -c:v libx264 -pix_fmt yuv420p -y {output_path}')
        if audio:
            output_path = fuse_audio_with_ffmpeg(input,output_path)
    clean_cache_file()
    # os.remove(input)
    return "Video Process Success, you can find the output in {} or click the download icon".format(output_path), output_path


def clean_cache_file(mask=False):
    temp_frames = os.listdir("./temp_frames")
    for i in temp_frames:
        os.remove(os.path.join("./temp_frames/{}".format(i)))
        if mask:
            os.remove(os.path.join("./temp_mask/{}".format(i)))
        os.remove(os.path.join("./temp_res_frames/{}".format(i)))