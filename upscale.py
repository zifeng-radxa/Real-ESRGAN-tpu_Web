import time
import os
from tpu_perf.infer import SGInfer
from tqdm import tqdm
import gradio as gr
import numpy as np
from util import fuse_audio_with_ffmpeg, resize_video

import cv2
del os.environ["LD_LIBRARY_PATH"]
from threading import Thread
import uuid
MODEL_PATH = './model/'
DEVICE_ID = 0


def load_model(model_name):
    model = EngineOV(model_path=os.path.join(MODEL_PATH, model_name), batch=1, device_id=DEVICE_ID)
    return model

class EngineOV:
    def __init__(self, model_path="", batch=1 ,device_id=0) :
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            print(">>>> device_id is in os.environ. and device_id = " ,device_id)
        self.model = SGInfer(model_path , batch=batch, devices=[device_id])

    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path ,self.device_id)

    def __call__(self, args):
        start = time.time()
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
            # print(values)
        task_id = self.model.put(*values)
        task_id, results, valid = self.model.get()
        # print(str(round((time.time() - start) * 1000, 3)) + " ms")
        return results

class Upscale():
    def __init__(self, input_path, output_path, model_name, tmp_path=None, type=None, num_worker=1):
        self.model_name = model_name
        self.type = type
        self.num_worker = int(num_worker)
        self.worker = []
        self.frames = []
        self.input_path = input_path
        self.output_path = output_path
        self.tmp_path = tmp_path
        self.init_worker()

        self.video_info = {}
        # self.video_writer, self.cap = self.get_video_info
        # self.frames_even = []
        self.inference_frames = []
        self.thread = []

    def init_worker(self):
        for i in range(self.num_worker):
            model = load_model(self.model_name)
            self.worker.append(model)
            self.frames.append([])

    def clean_cache(self):
        for i in self.frames:
            i.clear()

        self.inference_frames.clear()
        self.thread.clear()

    def model_inference(self, model, frame, tqdm_tool):
        for i in frame:
            res = model([np.expand_dims(i['data'], axis=0)])[0]

            # res = model([np.array([i["data"]])])[0]
            # 将图像从 chw 转换回 hwc
            frame_hwc = np.transpose(res, (0, 2, 3, 1))[0] * 255.0
            # 数据类型转回 uint8（为了写入到视频中）
            frame_hwc = np.clip(frame_hwc, 0, 255).astype(np.uint8)
            # 将处理后的帧写入到输出视频
            self.inference_frames.append({"id": i["id"], "data": frame_hwc})
            tqdm_tool.update(1)

        # print(t_name + " inference finish")

    def ratio_resize(self, img):
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


    def __call__(self, audio_check):
        if self.type == "video":
            self.cap = cv2.VideoCapture(self.input_path)
            if not self.cap.isOpened():
                print("Error: Couldn't open the video file.")
                raise gr.Error("Couldn't open the video file.")

            self.video_info['width'] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_info['height'] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # print(self.video_info['width'])
            # print(self.video_info['height'])
            if self.video_info['width'] != 640 or self.video_info['height'] != 480:
                self.cap.release()
                self.input_path = resize_video(self.input_path)
                self.cap = cv2.VideoCapture(self.input_path)
                if not self.cap.isOpened():
                    print("Error: Couldn't open the video file.")
                    raise gr.Error("Couldn't open the video file.")
                self.video_info['width'] = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.video_info['height'] = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            self.video_info['fps'] = self.cap.get(cv2.CAP_PROP_FPS)
            self.video_info['all_frame_num'] = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.video_writer = cv2.VideoWriter(self.tmp_path,
                                                fourcc,
                                                self.video_info['fps'],
                                                (self.video_info['width'] * 4, self.video_info['height'] * 4))

            SPLIT = 200
            print(self.video_info)
            total_task = self.video_info['all_frame_num'] // SPLIT + 1
            left_frame = self.video_info['all_frame_num'] % SPLIT

            each_task_frame = [SPLIT for _ in range(total_task - 1)]
            each_task_frame.append(left_frame)
            print("This would split {} tasks".format(total_task))

            for task_frame in each_task_frame:
                self.clean_cache()
                tqdm_tool = tqdm(total=task_frame)
                start = time.time()
                for i in range(task_frame):
                # while True:
                    ret, frame = self.cap.read()
                    # 如果视频读取完毕，跳出循环
                    if not ret:
                        break
                    # 数据类型转换为 float32
                    frame = frame.astype(np.float32)
                    # 将图像从 hwc 转换为 chw
                    frame_chw = np.transpose(frame, (2, 0, 1))
                    # 例如，这里你可以进行其他处理
                    frame_chw = frame_chw / 255.0
                    worker = i % self.num_worker
                    self.frames[worker].append({"id": i, "data": frame_chw})
                    tqdm_tool.update(1)
                tqdm_tool.close()
                print("frame ready")
                print(str(round((time.time() - start) * 1000, 3)) + " ms")



                tqdm_tool = tqdm(total=task_frame)
                start = time.time()
                for index, worker in enumerate(self.worker):
                    t = Thread(target=self.model_inference, args=(worker, self.frames[index], tqdm_tool))
                    self.thread.append(t)
                for i in self.thread:
                    i.start()

                for i in self.thread:
                    i.join()
                tqdm_tool.close()
                print(str(round((time.time() - start) * 1000, 3)) + " ms")



                tqdm_tool = tqdm(total=task_frame)
                start = time.time()
                self.inference_frames = sorted(self.inference_frames, key=lambda x: x["id"])

                for i in self.inference_frames:
                    self.video_writer.write(i["data"])
                    tqdm_tool.update(1)
                    # self.tqdm.update(1)
                # self.tqdm.close()
                # 释放视频对象
                tqdm_tool.close()
                print(str(round((time.time() - start) * 1000, 3)) + " ms")

            self.cap.release()
            self.video_writer.release()
            self.worker.clear()

            if audio_check:
                fuse_audio_with_ffmpeg(self.input_path, self.tmp_path, self.output_path)
                return self.output_path
            else:
                return self.tmp_path


        elif self.type == 'image':
            img = cv2.imread(self.input_path)
            h, w = img.shape[0:2]
            pad_black = None
            if h != 480 or w != 640:
                img, pad_black = self.ratio_resize(img)

            img = img.astype(np.float32)
            # 将图像从 hwc 转换为 chw
            frame_chw = np.transpose(img, (2, 0, 1))
            # 例如，这里你可以进行其他处理
            frame_chw = frame_chw / 255.0
            self.frames[0].append({"id": 1, "data": frame_chw})
            tqdm_tool = tqdm(total=1)
            self.model_inference(self.worker[0], self.frames[0], tqdm_tool=tqdm_tool)
            if pad_black:
                self.inference_frames[0]['data'] = self.inference_frames[0]['data'][pad_black[0]*4:1920-pad_black[1]*4, pad_black[2]*4:2560-pad_black[3]*4, :]

            cv2.imwrite(self.output_path, self.inference_frames[0]['data'])
            self.clean_cache()
            self.worker.clear()


            return self.output_path





