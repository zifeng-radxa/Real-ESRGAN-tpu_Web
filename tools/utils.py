import urllib.request
import os
import uuid
import cv2
import time
import socket
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse

# id_color_dict = {
#     # BGR
#     "White": [255, 255, 255],
#     "Blue": [219, 142, 67],
#     "Red": [0, 0, 255]
# }

id_color_dict = {
    # BGR
    "White": [255, 255, 255],
    "Blue": [67, 142, 219],
    "Red": [255, 0, 0]
}
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

def fuse_audio_with_ffmpeg(source_video_path, output_video_path):
    # os.environ["LD_LIBRARY_PATH"] = "/opt/sophon/libsophon-current/lib:"

    # ffmpeg -i original_video.mp4 -q:a 0 -map a temp_audio.mp3
    # ffmpeg -i test_half.mp4 -vn -acodec copy audioA.aac
    os.system("ffmpeg -y -i {} -vn -acodec copy ./result/video/audioA.aac".format(source_video_path))
    # ffmpeg -i input_video.mp4 -i input_audio.mp3 -c:v copy -c:a aac  -map 0:v:0 -map 1:a:0 -shortest output_combined.mp4
    os.system("ffmpeg -y -i {} -i ./result/video/audioA.aac -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 -shortest -y {}".format(output_video_path, output_video_path[:-4] + "_audio.mp4"))

    os.remove('./result/video/audioA.aac')
    os.remove(output_video_path)

    return output_video_path[:-4] + "_audio.mp4"
    # del os.environ["LD_LIBRARY_PATH"]


def resize_video(input_path):
    output_path = './result/resize_{}.mp4'.format(uuid.uuid4())
    # ffmpeg -c:v h264 -i aaaa.mp4 -vf "scale_bm=640:480:opt=2:zero_copy=1" -c:v h264_bm -y ff.mp4
    # os.environ["LD_LIBRARY_PATH"] = "/opt/sophon/libsophon-current/lib:"

    os.system('ffmpeg -c:v h264 -i {} -vf "scale_bm=640:480:opt=2:zero_copy=1" -c:v h264_bm -y {}'.format(input_path, output_path))
    #del os.environ["LD_LIBRARY_PATH"]

    return output_path


def ratio_resize(img, target_size):
    # target_size = (480, 640)
    old_size = img.shape[0:2]
    ratio = min(float(target_size[i]) / (old_size[i]) for i in range(len(old_size)))
    new_size = tuple([int(i * ratio) for i in old_size])
    img = cv2.resize(img, new_size[::-1], interpolation=cv2.INTER_LINEAR)
    pad_w = target_size[1] - new_size[1]
    pad_h = target_size[0] - new_size[0]
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    img_new = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    pad_black = (top, bottom, left, right)
    return img_new, pad_black

def get_model_list(mode_type="image"):
    files = os.listdir('./model/{}'.format(mode_type))
    model_list = []
    for i in files:
        if i[:4].lower() == "real":
            model_list.append(i)

    return model_list

def timer(func):
    def wrapper(*args, **kwargs):
        time0 = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - time0) * 1000
        print(f"Function '{func.__name__}' executed in: {elapsed:.2f} ms")
        return result
    return wrapper

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def hex_to_rgb(hex_color):
    """
    Convert HEX color to RGB.

    Args:
    hex_color (str): The HEX color string (e.g., '#ffffff').

    Returns:
    list: A list of three integers representing the RGB color (e.g., [255, 255, 255]).
    """
    # Remove the '#' character if present
    hex_color = hex_color.lstrip('#')

    # Convert the hex string to integers for R, G, and B
    rgb = [int(hex_color[i:i + 2], 16) for i in (0, 2, 4)]
    return rgb

def make_same_size(main_img, bg_img):
    main_h, main_w = main_img.shape[:2]
    bg_h, bg_w = bg_img.shape[:2]
    scale_w = main_w / bg_w
    scale_h = main_h / bg_h
    scale = max(scale_w, scale_h)

    new_w = int(bg_w * scale)
    new_h = int(bg_h * scale)
    resized_sub_image = cv2.resize(bg_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    start_x = (new_w - main_w) // 2
    start_y = (new_h - main_h) // 2
    cropped_sub_image = resized_sub_image[start_y:start_y + main_h, start_x:start_x + main_w, :]
    return cropped_sub_image


def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file
