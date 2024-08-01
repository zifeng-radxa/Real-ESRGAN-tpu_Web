import os
import uuid
import cv2
import numpy as np

from tools.utils import get_model_list, timer, fuse_audio_with_ffmpeg, id_color_dict, hex_to_rgb, make_same_size
from tools.tpu_utils import clean_tpu_memory
from core.imageupscaler import ImageUpscaler, ImageUpscaler2
from plugin.bgremove import Bgremover, Bgremover2

image_model_list = get_model_list()
video_model_list = get_model_list("video")
image_upscaler = ImageUpscaler()
image_upscaler2 = ImageUpscaler2()
bger = Bgremover()
bger2 = Bgremover2()

FFMPEG_CMD = "ffmpeg"

@timer
def image_pipeline(input, model, face_enhance=None, background_remove=None, bg_color="White", user_bg_color=None, bg_img=None, bg_img_up=None, output_path=None, save=False):
    clean_tpu_memory([image_upscaler2, bger2])
    if input is None:
        return ("Please upload image", None)
    bg_color_value = None
    if bg_img is None:
        if bg_color == "Pick color":
            bg_color_value = hex_to_rgb(user_bg_color)
        elif bg_color != "Transparent":
            bg_color_value = id_color_dict[bg_color]
    if isinstance(input, str):
        img = cv2.imread(input)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif isinstance(input, np.ndarray):
        img = input

    if background_remove == 0:
        image_upscaler.change_model(model, face_enhance_name="None", bg_helper=True)
        bg_hlepr_img = image_upscaler.forward(img, face_enhance="None", bg_upscale=True)
        if model == "realesr-animevideo_v3_rgb_1_3_480_640.bmodel":
            res = bg_hlepr_img
        else:
            image_upscaler.change_model(model, face_enhance_name=face_enhance, bg_helper=False)
            res = image_upscaler.forward(img, face_enhance)
        if bg_img is not None and bg_img_up:
            bg_img = image_upscaler.forward(bg_img, face_enhance)
        bger.init_model()

        if bg_color_value is not None or bg_img is not None:
            _, mask = bger.forward(bg_hlepr_img, only_black_white=True)
            new_mask = cv2.resize(mask, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_LINEAR)
            res = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
            if bg_img is None:
                bg_img = np.zeros_like(res)
                bg_img[:, :, :] = bg_color_value
            else:
                bg_img = make_same_size(res, bg_img)

            bg_are = cv2.bitwise_and(bg_img, bg_img, mask=cv2.bitwise_not(new_mask))
            res = cv2.bitwise_and(res, res, mask=new_mask)
            res = cv2.add(bg_are, res)

        else:
            _, mask = bger.forward(bg_hlepr_img)
            new_mask = cv2.resize(mask, (res.shape[1], res.shape[0]), interpolation=cv2.INTER_LINEAR)
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGRA)
            res = res[:,:,[2,1,0,3]]
            res[:,:,3] = new_mask


    elif background_remove == 1:
        # only bgremove
        bger.init_model()
        if bg_color_value is not None or bg_img is not None:
            res, mask = bger.forward(img, only_black_white=True)
            if bg_img is None:
                bg_img = np.zeros_like(input)
                bg_img[:, :, :] = bg_color_value
            else:
                bg_img = make_same_size(res, bg_img)
            res = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
            bg_are = cv2.bitwise_and(bg_img, bg_img, mask=cv2.bitwise_not(mask))
            res = cv2.bitwise_and(res, res, mask=mask)
            res = cv2.add(bg_are, res)
        else:
            res, mask = bger.forward(img)

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
    clean_tpu_memory([image_upscaler, bger, image_upscaler2, bger2])
    if not isinstance(input, str) or input is None:
        print("Please upload your video")
        return "Please upload your video", None

    if input.endswith('.flv'):
        mp4_path = input.replace('.flv', '.mp4')
        os.system(f'{FFMPEG_CMD} -i {input} -codec copy {mp4_path}')
        input = mp4_path

    if not os.path.exists('./temp_frames'):
        os.makedirs('./temp_frames', exist_ok=True)
    if not os.path.exists('./temp_res_frames'):
        os.makedirs('./temp_res_frames', exist_ok=True)
    if not os.path.exists('./temp_mask'):
        os.makedirs('./temp_mask', exist_ok=True)

    cap = cv2.VideoCapture(input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print("**************")
    os.system(f'{FFMPEG_CMD} -i {input} -qscale:v 1 -qmin 1 -qmax 1 -vsync 0  ./temp_frames/frame%d.png')
    print("**************")

    if background_remove == 0:
        print(face_enhance)
        print(type(face_enhance))
        # upscale + rmgb + upscale
        print("up scaling video step 1")
        image_upscaler2.change_model("realesr-animevideov3_f16_video.bmodel", face_enhance, thread_num=thread)
        image_upscaler2.forward('./temp_frames', face_enhance=face_enhance)
        clean_tpu_memory([image_upscaler2])
        print("removing background")
        bger2.init_model(thread_num=thread)
        bger2.forward('./temp_res_frames', save_type="mask")
        if model == "realesr-animevideov3_f16_video.bmodel":
            print("merge background")
            bger2.forward('./temp_frames', call_back="run_io_bgrm_progress")
        else:
            clean_tpu_memory([bger2])
            image_upscaler2.change_model(model, face_enhance, thread_num=thread)
            print("up scaling video step 2")
            image_upscaler2.forward('./temp_frames')
            clean_tpu_memory([image_upscaler2])
            print("merge background")
            bger2.init_model(thread_num=thread)
            bger2.forward('./temp_frames', call_back="run_io_bgrm_progress")


    elif background_remove == 1:
        # only remove bg
        print("removing background")
        bger2.init_model(thread_num= thread)
        bger2.forward('./temp_frames', save_type="image")


    elif background_remove == 2:
        # only upscale
        print("up scaling video")
        image_upscaler2.change_model(model, face_enhance, thread_num=thread)
        image_upscaler2.forward('./temp_frames', face_enhance)


    if save:
        if output_path is None:
            if not os.path.exists(os.path.join('./result', 'video')):
                os.makedirs(os.path.join('./result', 'video'), exist_ok=True)
            output_path = './result/video/out_{}.mp4'.format(uuid.uuid4())
        print("merge")
        os.system(f'{FFMPEG_CMD} -framerate {fps} -i ./temp_res_frames/frame%d.png -c:v libx264 -pix_fmt yuv420p -y {output_path}')
        if audio:
            output_path = fuse_audio_with_ffmpeg(input,output_path)
    clean_cache_file(background_remove == 0)
    # os.remove(input)
    return "Video Process Success, you can find the output in {} or click the download icon".format(output_path), output_path


def clean_cache_file(mask=False):
    temp_frames = os.listdir("./temp_frames")
    for i in temp_frames:
        os.remove(os.path.join("./temp_frames/{}".format(i)))
        if mask:
            os.remove(os.path.join("./temp_mask/{}".format(i)))
        os.remove(os.path.join("./temp_res_frames/{}".format(i)))