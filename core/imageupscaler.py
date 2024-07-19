import time
import os
from tqdm import tqdm
import numpy as np
from threading import Thread
# from tools.writer import Writer
from tools.tpu_utils import load_bmodel
from tools.utils import ratio_resize
import cv2

try:
    del os.environ["LD_LIBRARY_PATH"]
except Exception as e:
    pass


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


    def __call__(self, input, face_enhance=None, bg_upscale=False):
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
