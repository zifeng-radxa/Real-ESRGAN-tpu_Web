import time
import os
from tqdm import tqdm
import numpy as np
from tools.tpu_utils import load_bmodel
from tools.utils import ratio_resize, timer
import cv2
from concurrent.futures import ThreadPoolExecutor
import asyncio


class ImageUpscaler():
    def __init__(self):
        self.cur_model_name = None
        self.cur_face_enhance_name = None

        self.re_upscale_model = None
        self.face_enhance_model = None
        self.face_detect_model = None
        self.face_pars_model = None
        self.bg_helper_model = None

    def change_model(self,model_name, face_enhance_name, bg_helper=False):
        if model_name != self.cur_model_name:
            self.re_upscale_model = load_bmodel(model_name)
            self.cur_model_name = model_name
        if face_enhance_name != "None" and face_enhance_name != self.cur_face_enhance_name:
            if face_enhance_name == "CodeFormer":
                face_enhance_bmodel = "codeformer_1-3-512-512_1-235ms.bmodel"
                self.face_enhance_model = load_bmodel(face_enhance_bmodel)
            elif face_enhance_name == "GFPGAN":
                face_enhance_bmodel = "gfpgan.bmodel"
                self.face_enhance_model = load_bmodel(face_enhance_bmodel)

            self.cur_face_enhance_name = face_enhance_name
            if self.face_detect_model is None:
                self.face_detect_model = load_bmodel("retinaface_resnet50_rgb_1_3_480_640.bmodel")
            if self.face_pars_model is None:
                self.face_pars_model = load_bmodel("parsing_parsenet_rgb_1_3_512_512.bmodel")

        if self.bg_helper_model is None and bg_helper:
            self.bg_helper_model = load_bmodel("realesr-animevideo_v3_rgb_1_3_480_640.bmodel")

    def clean_tpu_memory(self):
        self.cur_model_name = None
        self.cur_face_enhance_name = None
        self.re_upscale_model = None
        self.face_enhance_model = None
        self.face_detect_model = None
        self.face_pars_model = None
        self.bg_helper_model = None

    def model_inference(self, frame, bg_helper=False):
        if bg_helper:
            res = self.bg_helper_model([np.expand_dims(frame, axis=0)])[0][0]
        else:
            res = self.re_upscale_model([np.expand_dims(frame, axis=0)])[0][0]
        res = np.transpose(res, (1,2,0)) * 255.0
        # 将图像从 chw 转换回 hwc, rgb->brg
        # res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
        res = np.clip(res, 0, 255).astype(np.uint8)
        return res


    def forward(self, input, face_enhance=None, bg_upscale=False):
        h, w = input.shape[0:2]
        pad_black = None
        if h != 480 or w != 640:
            img, pad_black = ratio_resize(input, target_size=(480, 640))
        else:
            img = input
        img = img.astype(np.float32)

        if face_enhance != "None":
            tqdm_tool = tqdm(total=4)
            img_copy = img.copy()
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
            # 将图像从 hwc 转换为 chw
            frame_chw = np.transpose(img, (2, 0, 1))
            # 归一化
            frame_chw = frame_chw / 255.0
            res_frame = self.model_inference(frame_chw)
            tqdm_tool.update(1)

            from plugin.face_enhance import FaceEnhance
            face_enhancer = FaceEnhance(self.re_upscale_model, self.face_detect_model, self.face_pars_model, self.face_enhance_model, name=self.cur_face_enhance_name)
            res_frame = face_enhancer.run(img_copy, tqdm_tool, res_frame)

        elif bg_upscale:
            frame_chw = np.transpose(img, (2, 0, 1))
            # 归一化
            frame_chw = frame_chw / 255.0
            res_frame = self.model_inference(frame_chw, bg_helper=True)

        else:
            tqdm_tool = tqdm(total=1)
            # 将图像从 hwc 转换为 chw
            frame_chw = np.transpose(img, (2, 0, 1))
            # 归一化
            frame_chw = frame_chw / 255.0
            res_frame = self.model_inference(frame_chw)
            tqdm_tool.update(1)

        if pad_black is not None:
            res_frame = res_frame[pad_black[0]*4:1920-pad_black[1]*4, pad_black[2]*4:2560-pad_black[3]*4, :]

        return res_frame


class ImageUpscaler2(ImageUpscaler):
    def __init__(self, thread_num=0):
        super().__init__()
        self.nets = []
        self.thread_num = thread_num
        self.img_shape = None
        self.pad_black = None

    def clean_tpu_memory(self):
        self.nets.clear()
        self.thread_num = 0
        self.img_shape = None
        self.pad_black = None
        self.cur_model_name = None
        self.cur_face_enhance_name = None
        self.re_upscale_model = None
        self.face_enhance_model = None
        self.face_detect_model = None
        self.face_pars_model = None
        self.bg_helper_model = None

    def change_model(self, model_name, face_enhance_name, bg_helper=False, thread_num=1):
        # print("model_name: {}".format(model_name))
        if model_name != self.cur_model_name:
            self.nets.clear()
            for i in range(1):
                net = load_bmodel(model_name, model_type="video")
                self.nets.append(net)
            self.thread_num = thread_num
            self.cur_model_name = model_name

        if self.thread_num != thread_num:
            diff = 1 - len(self.nets)
            # print("diff: {}".format(diff))
            if diff > 0:
                for i in range(diff):
                    net = load_bmodel(model_name, model_type="video")
                    self.nets.append(net)
            else:
                for i in range(-diff):
                    self.nets.pop()
            self.thread_num = thread_num

        if face_enhance_name != "None" and face_enhance_name != self.cur_face_enhance_name:
            if face_enhance_name == "CodeFormer":
                face_enhance_bmodel = "codeformer_1-3-512-512_1-235ms.bmodel"
                self.face_enhance_model = load_bmodel(face_enhance_bmodel)
            elif face_enhance_name == "GFPGAN":
                face_enhance_bmodel = "gfpgan.bmodel"
                self.face_enhance_model = load_bmodel(face_enhance_bmodel)

            self.cur_face_enhance_name = face_enhance_name
            if self.face_detect_model is None:
                self.face_detect_model = load_bmodel("retinaface_resnet50_rgb_1_3_480_640.bmodel")
            if self.face_pars_model is None:
                self.face_pars_model = load_bmodel("parsing_parsenet_rgb_1_3_512_512.bmodel")

        if self.bg_helper_model is None and bg_helper:
            self.bg_helper_model = load_bmodel("realesr-animevideo_v3_rgb_1_3_480_640.bmodel")



    def preprocess(self, img, normalization=True):
        if self.img_shape[0] != 480 or self.img_shape[1] != 640:
            img, self.pad_black = ratio_resize(img, target_size=(480, 640))
        if normalization:
            img = img.astype(np.float32)
            img = np.transpose(img, (2, 0, 1))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)

        return img

    def postprocess(self, img, pad=True):
        img = np.transpose(img, (1,2,0)) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        if pad:
            # if self.pad_black is not None:
            img = img[self.pad_black[0]*4:1920-self.pad_black[1]*4, self.pad_black[2]*4:2560-self.pad_black[3]*4, :]
        return img

    async def async_imwrite(self, path, img):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, cv2.imwrite, path, img)

    async def image_upscale_thread(self, res_frame, target_path, file_name, async_task):
        img_data = self.postprocess(res_frame)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR)
        task = self.async_imwrite(os.path.join(target_path, file_name), img_data)
        async_task.append(task)

    async def face_enhance_thread(self, res_frame, img_copy, target_path, file_name, async_task):
        print("in")
        up_img = self.postprocess(res_frame, pad=False)
        from plugin.face_enhance import FaceEnhance
        face_enhancer = FaceEnhance(self.re_upscale_model, self.face_detect_model, self.face_pars_model,
                                    self.face_enhance_model, name=self.cur_face_enhance_name)
        res_frame = face_enhancer.run(img_copy, None, up_img)

        # res_frame = self.postprocess(res_frame, pad=False)
        res_frame = cv2.cvtColor(res_frame, cv2.COLOR_RGBA2BGR)
        res_frame = res_frame[self.pad_black[0] * 4:1920 - self.pad_black[1] * 4,
                    self.pad_black[2] * 4:2560 - self.pad_black[3] * 4, :]

        task = self.async_imwrite(os.path.join(target_path, file_name), res_frame)
        async_task.append(task)

    def run_face_enhance_thread(self, res_frame, img_copy, target_path, file_name, async_task):
        asyncio.run(self.face_enhance_thread(res_frame, img_copy, target_path, file_name, async_task))

    def run_image_upscale_thread(self, res_frame, target_path, file_name, async_task):
        asyncio.run(self.image_upscale_thread(res_frame, target_path, file_name, async_task))

    async def thread_progress(self, input_file_name, worker, source_path=None, target_path='./temp_res_frames', face_enhance="None"):
        imgs_data = [cv2.imread(os.path.join(source_path, i),) for i in input_file_name]
        imgs_infer_data = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in imgs_data]
        imgs_infer_data = [self.preprocess(i) for i in imgs_infer_data]
        if face_enhance != "None":
            imgs_copy = [self.preprocess(i, normalization=False) for i in imgs_data]
        imgs_data.clear()
        input_numpy = np.concatenate(imgs_infer_data, axis=0)
        imgs_infer_data.clear()
        res_frames = worker([input_numpy])[0]

        async_task = []
        with ThreadPoolExecutor(4) as t:
            for i in range(len(res_frames)):
                if face_enhance != "None":
                    t.submit(self.run_face_enhance_thread, res_frames[i], imgs_copy[i], target_path, input_file_name[i], async_task)
                else:
                    t.submit(self.run_image_upscale_thread, res_frames[i], target_path, input_file_name[i],
                             async_task)
            del res_frames

        await asyncio.gather(*async_task)

    def run_thread_progress(self, input_file_name, worker, source_path, face_enhance, tqdm_tool=None):
        asyncio.run(self.thread_progress(input_file_name, worker, source_path, face_enhance=face_enhance))
        tqdm_tool.update(1)


    def get_image_meta(self, input):
        img = cv2.imread(input)
        self.img_shape = img.shape[:2]

    def forward(self, input, face_enhance=None, bg_upscale=False):
        if isinstance(input, str):
            imgs_file = os.listdir(input)
            total_img_num = len(imgs_file)
            self.get_image_meta(os.path.join(input, imgs_file[0]))
            SPLIT = 30
            task_num = total_img_num // SPLIT if total_img_num % SPLIT == 0 else total_img_num // SPLIT + 1
            frame_split = [SPLIT for _ in range(task_num - 1)]
            frame_split.append(SPLIT if total_img_num % SPLIT == 0 else total_img_num % SPLIT)
            # print(frame_split)
            tqdm_tool = tqdm(total=task_num)
            print("thread_num: {}".format(self.thread_num))
            with ThreadPoolExecutor(self.thread_num) as t:
                for i in range(0, task_num):
                    sub_img_file = imgs_file[i * SPLIT:(i * SPLIT + SPLIT)]
                    t.submit(self.run_thread_progress,
                             sub_img_file, self.nets[0], input, face_enhance, tqdm_tool)





