import time
import os
from tpu_perf.infer import SGInfer
from tqdm import tqdm
import gradio as gr
import numpy as np
from util import fuse_audio_with_ffmpeg, resize_video
import cv2
from threading import Thread
# from tools.writer import Writer
import uuid
from tools.tpu_utils import load_model

try:
    del os.environ["LD_LIBRARY_PATH"]
except Exception as e:
    pass


class Upscale():
    def __init__(self, input_path, output_path, model_name, tmp_path=None, type=None, face_enhance=None, num_worker=1):
        self.model_name = model_name
        self.type = type
        self.num_worker = int(num_worker)
        self.worker = []
        self.frames = []
        self.input_path = input_path
        self.output_path = output_path
        self.tmp_path = tmp_path
        self.face_enhance = face_enhance
        self.inference_frames = []
        self.thread = []
        self.face_models = []
        self.pars_models = []
        self.video_info = {}
        self.init_worker()


        # self.video_writer, self.cap = self.get_video_info
        # self.frames_even = []


    def init_worker(self):
        for i in range(self.num_worker):
            model = load_model(self.model_name)
            self.worker.append(model)
            self.frames.append([])
        if self.face_enhance:
            self.init_face_pars_model()
    def init_face_pars_model(self):
        for i in range(self.num_worker):
            model_reretinaface_resnet50 = load_model('retinaface_resnet50_rgb_1_3_480_640.bmodel')
            self.face_models.append(model_reretinaface_resnet50)

        for i in range(self.num_worker):
            model_parsing_parsenet = load_model('parsing_parsenet_rgb_1_3_512_512.bmodel')
            self.pars_models.append(model_parsing_parsenet)

        # self.fa

    def clean_cache(self):
        for i in self.frames:
            i.clear()

        self.inference_frames.clear()
        self.thread.clear()

    def model_inference(self, model, frame, tqdm_tool):
        for i in frame:
            res = model([np.expand_dims(i['data'], axis=0)])[0]
            # res = model([np.array([i["data"]])])[0]
            # 将图像从 chw 转换回 hwc, rgb->brg
            frame_hwc = np.transpose(res[:, [2, 1, 0], :, :], (0, 2, 3, 1))[0] * 255.0
            # 数据类型转回 uint8（为了写入到视频中）
            frame_hwc = np.clip(frame_hwc, 0, 255).astype(np.uint8)
            # frame_hwc = cv2.cvtColor(frame_hwc, cv2.COLOR_RGB2BGR)
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

            # self.writer = Writer(audio_check, self.output_path, self.tmp_path, self.video_info['fps'])
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            self.video_writer = cv2.VideoWriter(self.tmp_path,
                                                fourcc,
                                                self.video_info['fps'],
                                                (self.video_info['width'] * 4, self.video_info['height'] * 4))

            SPLIT = 50
            print(self.video_info)
            total_task = self.video_info['all_frame_num'] // SPLIT + 1
            left_frame = self.video_info['all_frame_num'] % SPLIT

            each_task_frame = [SPLIT for _ in range(total_task - 1)]
            each_task_frame.append(left_frame)
            print("This would split {} tasks".format(total_task))

            for index, task_frame in enumerate(each_task_frame):
                self.clean_cache()
                print("TASK: {}".format(index))
                print("capture frame: ")
                tqdm_tool = tqdm(total=task_frame)
                start = time.time()
                for i in range(task_frame):
                    ret, frame = self.cap.read()
                    # 如果视频读取完毕，跳出循环
                    if not ret:
                        break
                    # 数据类型转换为 float32
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32)
                    # 将图像从 hwc 转换为 chw, bgr->rgb
                    frame_chw = np.transpose(frame[:, :, [2, 1, 0]], (2, 0, 1))
                    # 例如，这里你可以进行其他处理
                    frame_chw = frame_chw / 255.0
                    worker = i % self.num_worker
                    self.frames[worker].append({"id": i, "data": frame_chw})
                    tqdm_tool.update(1)
                tqdm_tool.close()
                print(str(round((time.time() - start) * 1000, 3)) + " ms")


                print("frame inference: ")
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

                # use ffmpeg
                # tqdm_tool = tqdm(total=task_frame)
                # start = time.time()
                # print("encode frame: ")
                # self.inference_frames = sorted(self.inference_frames, key=lambda x: x["id"])
                # for i in self.inference_frames:
                #     self.writer.write_frame(i["data"])
                #     # tqdm_tool.update(1)
                #     # self.tqdm.update(1)
                # # self.tqdm.close()
                # # 释放视频对象
                # # tqdm_tool.close()
                # print(str(round((time.time() - start) * 1000, 3)) + " ms")

                # use opencv encode
                print("encode frame: ")
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
            # self.writer.close()
            self.worker.clear()

            if audio_check:
                fuse_audio_with_ffmpeg(self.input_path, self.tmp_path, self.output_path)
                return self.output_path
            else:
                return self.tmp_path


        elif self.type == 'image':
            if self.face_enhance:
                img = cv2.imread(self.input_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[0:2]
                pad_black = None
                if h != 480 or w != 640:
                    img, pad_black = self.ratio_resize(img)

                self.frames[0].append({"id": 1, "data": img})
                tqdm_tool = tqdm(total=3)

                from face_enhance import FaceEnhance
                face_enhancer = FaceEnhance(self.worker[0], self.face_models[0], self.pars_models[0])
                output = face_enhancer.run(self.frames[0][0]['data'], tqdm_tool)
                if pad_black:
                    output = output[pad_black[0]*4:1920-pad_black[1]*4, pad_black[2]*4:2560-pad_black[3]*4, :]
                tqdm_tool.update(1)

                cv2.imwrite(self.output_path, output)

            else:
                img = cv2.imread(self.input_path)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[0:2]
                pad_black = None
                if h != 480 or w != 640:
                    img, pad_black = self.ratio_resize(img)

                img = img.astype(np.float32)
                # 将图像从 hwc 转换为 chw, bgr -> rgb
                frame_chw = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
                # 例如，这里你可以进行其他处理
                frame_chw = frame_chw / 255.0
                self.frames[0].append({"id": 1, "data": frame_chw})
                tqdm_tool = tqdm(total=1)
                # if not face_enhance:
                self.model_inference(self.worker[0], self.frames[0], tqdm_tool=tqdm_tool)


                if pad_black:
                    self.inference_frames[0]['data'] = self.inference_frames[0]['data'][pad_black[0]*4:1920-pad_black[1]*4, pad_black[2]*4:2560-pad_black[3]*4, :]

                cv2.imwrite(self.output_path, self.inference_frames[0]['data'])

            self.clean_cache()
            self.worker.clear()
            self.face_models.clear()
            self.pars_models.clear()

            return self.output_path
