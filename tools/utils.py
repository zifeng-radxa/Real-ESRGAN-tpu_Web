import urllib.request
import os
import uuid
import cv2

def download_file(url, file_name, folder_path='model'):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
        print(f"File downloaded and saved to: {file_path}")

def export_envs():
    envs = {
        "LD_LIBRARY_PATH": "/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH",
        "PYTHONPATH": "/opt/sophon/sophon-opencv-latest/opencv-python/:$PYTHONPATH"
    }
    for key, value in envs.items():
        if key in os.environ and os.environ[key] == value:
            continue
        else:
            os.environ[key] = value
            print("export {}={}".format(key, value))

    for key, value in os.environ.items():
        print(f"{key}: {value}")

def fuse_audio_with_ffmpeg(source_video_path, output_video_path, output_video_with_audio):
    os.environ["LD_LIBRARY_PATH"] = "/opt/sophon/libsophon-current/lib:"

    # ffmpeg -i original_video.mp4 -q:a 0 -map a temp_audio.mp3
    os.system("ffmpeg -y -i {} -q:a 4 -map a ./result/temp_audio.mp3".format(source_video_path))
    # ffmpeg -i input_video.mp4 -i input_audio.mp3 -c:v copy -c:a aac  -map 0:v:0 -map 1:a:0 -shortest output_combined.mp4
    os.system("ffmpeg -y -i {} -i ./result/temp_audio.mp3 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest {}".format(output_video_path, output_video_with_audio))

    del os.environ["LD_LIBRARY_PATH"]


def resize_video(input_path):
    output_path = './result/resize_{}.mp4'.format(uuid.uuid4())
    # ffmpeg -c:v h264 -i aaaa.mp4 -vf "scale_bm=640:480:opt=2:zero_copy=1" -c:v h264_bm -y ff.mp4
    os.environ["LD_LIBRARY_PATH"] = "/opt/sophon/libsophon-current/lib:"

    os.system('ffmpeg -c:v h264 -i {} -vf "scale_bm=640:480:opt=2:zero_copy=1" -c:v h264_bm -y {}'.format(input_path, output_path))
    del os.environ["LD_LIBRARY_PATH"]

    return output_path


def ratio_resize(img):
    target_size = (480, 640)
    old_size = img.shape[0:2]
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])
    img = cv2.resize(img, new_size[::-1])
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    pad_black = (top, bottom, left, right)
    return img_new, pad_black

def get_model_list():
    files = os.listdir('./model')
    model_list = []
    for i in files:
        if i[:4].lower() == "real":
            model_list.append(i)

    return model_list