import time
import os
from tpu_perf.infer import SGInfer
from tqdm import tqdm
import gradio as gr
import numpy as np
from tools.utils import fuse_audio_with_ffmpeg, resize_video
import cv2
from threading import Thread
# from tools.writer import Writer
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
        self.enhance_models = []
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
            model_reretinaface_resnet50 = load_model('retinaface_resnet50_rgb_1684x_BF16.bmodel')
            self.face_models.append(model_reretinaface_resnet50)

        for i in range(self.num_worker):
            model_parsing_parsenet = load_model('parsing_parsenet_rgb_1684x_BF16.bmodel')
            self.pars_models.append(model_parsing_parsenet)

        for i in range(self.num_worker):
            model_enhance = load_model('codeformer_1-3-512-512_1-235ms.bmodel')
            self.enhance_models.append(model_enhance)

        # self.fa

    def clean_cache(self):
        for i in self.frames:
            i.clear()

        self.inference_frames.clear()
        self.thread.clear()

    def model_inference(self, model, frame, tqdm_tool):
        for i in range(0, len(frame), 3):
            n_frame_list = []
            n_id = []
            for i in range(3):
                if frame:
                    _frame = frame.pop()
                    n_frame_list.append(_frame["data"])
                    n_id.append(_frame["id"])
            # if len(n_frame) == 0:
            #     continue
            # print(len(n_frame_list))
            n_frame = np.stack(n_frame_list)
            # print(n_frame.shape)
            del n_frame_list
            # res = model([np.expand_dims(i['data'], axis=0)])[0]
            # print(i['data'].shape)
            time0 = time.time()
            res = model([n_frame])[0]
            print("model infrence: {} ms".format((time.time() - time0) * 1000))
            # print(res.shape)
            # res = model([np.array([i["data"]])])[0]
            # 将图像从 chw 转换回 hwc, rgb->brg
            frame_hwc = np.transpose(res[:, [2, 1, 0], :, :], (0, 2, 3, 1)) * 255.0
            # 数据类型转回 uint8（为了写入到视频中）
            frame_hwc = np.clip(frame_hwc, 0, 255).astype(np.uint8)
            # frame_hwc = cv2.cvtColor(frame_hwc, cv2.COLOR_RGB2BGR)
            # 将处理后的帧写入到输出视频
            for i in range(len(n_id)):
                self.inference_frames.append({"id": n_id[i], "data": frame_hwc[i]})
            tqdm_tool.update(len(n_id))

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
            print(each_task_frame)
            for index, task_frame in enumerate(each_task_frame):
                self.clean_cache()
                print("TASK: {}".format(index))

                if self.face_enhance:
                    from face_enhance import FaceEnhance
                    tqdm_tool = tqdm(total=task_frame)
                    start = time.time()
                    for i in range(task_frame):
                        ret, frame = self.cap.read()
                        # 如果视频读取完毕，跳出循环
                        if not ret:
                            break
                        worker = i % self.num_worker
                        self.frames[worker].append({"id": i, "data": frame})
                        tqdm_tool.update(1)
                    tqdm_tool.close()
                    print("capture frame: {} ms".format(round((time.time() - start) * 1000, 3)))

                    tqdm_tool = tqdm(total=task_frame)
                    start = time.time()
                    for index, worker in enumerate(self.worker):
                        face_enhancer = FaceEnhance(self.worker[index], self.face_models[index], self.pars_models[index],
                                                    self.enhance_models[index])

                        t = Thread(target=face_enhancer.run, args=(self.frames[index], self.inference_frames, tqdm_tool))
                        self.thread.append(t)
                    for i in self.thread:
                        i.start()

                    for i in self.thread:
                        i.join()
                    tqdm_tool.close()
                    print("frame inference: {} ms".format(round((time.time() - start) * 1000, 3)))

                    tqdm_tool = tqdm(total=task_frame)
                    start = time.time()
                    self.inference_frames = sorted(self.inference_frames, key=lambda x: x["id"], reverse=True)
                    for _ in range(len(self.inference_frames)):
                        self.video_writer.write(self.inference_frames.pop()["data"])
                        tqdm_tool.update(1)
                        # self.tqdm.update(1)
                    # self.tqdm.close()
                    # 释放视频对象
                    tqdm_tool.close()
                    print("encode frame: {} ms".format(round((time.time() - start) * 1000, 3)))

                else:
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
                    print("\ncapture frame: {} ms\n".format(round((time.time() - start) * 1000, 3)))


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
                    print("\nframe inference: {} ms\n".format(round((time.time() - start) * 1000, 3)))



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

                    tqdm_tool = tqdm(total=task_frame)
                    start = time.time()
                    self.inference_frames = sorted(self.inference_frames, key=lambda x: x["id"], reverse=True)
                    for _ in range(len(self.inference_frames)):
                        self.video_writer.write(self.inference_frames.pop()["data"])
                        tqdm_tool.update(1)
                        # self.tqdm.update(1)
                    # self.tqdm.close()
                    # 释放视频对象
                    tqdm_tool.close()
                    print("encode frame: {} ms".format(round((time.time() - start) * 1000, 3)))

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
                tqdm_tool = tqdm(total=4)

                from face_enhance import FaceEnhance
                face_enhancer = FaceEnhance(self.worker[0], self.face_models[0], self.pars_models[0], self.enhance_models[0])
                face_enhancer.run(self.frames[0][0], self.inference_frames, tqdm_tool)
                if pad_black:
                    self.inference_frames[0]['data'] = self.inference_frames[0]['data'][pad_black[0]*4:1920-pad_black[1]*4, pad_black[2]*4:2560-pad_black[3]*4, :]
                tqdm_tool.update(1)

                cv2.imwrite(self.output_path, self.inference_frames[0]['data'])

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
                tqdm_tool = tqdm(total=1)
                # test_a = np.zeros((3,3,480,640)).astype(np.float32)
                # test_a[:1] = frame_chw
                # print(test_a[0])
                # test_a = np.stack([frame_chw, frame_chw, frame_chw])
                # test_a = np.concatenate((frame_chw, frame_chw, frame_chw), axis=0)
                # test_a = np.concatenate((test_a, frame_chw), axis=0)

                self.frames[0].append({"id": 1, "data": frame_chw})

                # if not face_enhance:
                self.model_inference(self.worker[0], self.frames[0], tqdm_tool=tqdm_tool)


                if pad_black:
                    print(self.inference_frames[0]['data'].shape)
                    self.inference_frames[0]['data'] = self.inference_frames[0]['data'][pad_black[0]*4:1920-pad_black[1]*4, pad_black[2]*4:2560-pad_black[3]*4, :]

                cv2.imwrite(self.output_path, self.inference_frames[0]['data'])

            self.clean_cache()
            self.worker.clear()
            self.face_models.clear()
            self.pars_models.clear()

            return self.output_path
