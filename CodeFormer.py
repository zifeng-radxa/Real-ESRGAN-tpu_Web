import cv2
import os
import torch
from basicsr.utils import img2tensor, tensor2img
from FACEXLIB.facexlib.utils.face_restoration_helper import FaceRestoreHelper
from torchvision.transforms.functional import normalize

import numpy as np
import time

class FaceRestorerCodeFormer():
    def name(self):
        return "CodeFormer"

    def __init__(self, code_bmodel=None, face_bmodel=None, pars_bmodel=None, bg_upsampler=None):
        self.net = code_bmodel
        self.face_helper = None
        self.bg_upsampler = bg_upsampler

        self.face_helper = FaceRestoreHelper(
            4,
            face_size=512,
            crop_ratio=(1, 1),
            det_model='retinaface_resnet50',
            save_ext='png',
            use_parse=True,
            face_bmodel=face_bmodel,
            pars_bmodel=pars_bmodel
        )

    # def create_models(self, bmodel_path):  # ckpt_path='./weights/codeformer-v0.1.0.pth'):
    #     if self.net is not None and self.face_helper is not None:
    #         return self.net, self.face_helper
    #     face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50',
    #                                     save_ext='png', use_parse=True)  # , device=devices.device_codeformer)
    #     net = EngineOV(model_path=bmodel_path, device_id=0)
    #     self.net = net
    #     self.face_helper = face_helper
    #     return net, face_helper

    # def enhance(self, np_image, w=None):
    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, w=0.5, tqdm_tool=None):
        # img => BGR
        self.face_helper.clean_all()

        # np_image = np_image[:, :, ::-1]

        original_resolution = img.shape[0:2]

        # if self.net is None or self.face_helper is None:
        #     return np_image
        self.face_helper.read_image(img)
        time0 = time.time()
        self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
        self.face_helper.align_warp_face()

        tqdm_tool.update(1)
        print("face detect: {}".format((time.time() - time0) * 1000))


        for cropped_face in self.face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0)  # .to(devices.device_codeformer)
            try:
                time0 = time.time()
                with torch.no_grad():  # shared.opts.code_former_weight
                    # in: (1, 3, 512, 512)    out: (1, 3, 512, 512) (1, 256, 1024) (1, 256, 16, 16)
                    output = \
                    self.net([cropped_face_t.numpy(), np.array([w if w is not None else 0.5], dtype=np.float32)])[
                        0]  ## the dtype must be explicitly set
                    restored_face = tensor2img(torch.from_numpy(output).squeeze(0), rgb2bgr=True, min_max=(-1, 1))
                del output
                print("face enhance: {}".format((time.time() - time0) * 1000))

            except Exception:
                print('Failed inference for CodeFormer')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        tqdm_tool.update(1)

        if not has_aligned and paste_back:
            # upsample the background
            if self.bg_upsampler is not None:
                img = img.astype(np.float32)
                frame_chw = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
                frame_chw = frame_chw / 255.0

                # Now only support RealESRGAN for upsampling background
                time0 = time.time()
                print(frame_chw.shape)
                res = self.bg_upsampler([np.expand_dims(frame_chw, axis=0)])[0]
                print("bg upsampler: {}".format((time.time() - time0) * 1000))
                tqdm_tool.update(1)

                # res = model([np.array([i["data"]])])[0]
                # 将图像从 chw 转换回 hwc, rgb->brg
                frame_hwc = np.transpose(res[:, [2, 1, 0], :, :], (0, 2, 3, 1))[0] * 255.0
                # 数据类型转回 uint8（为了写入到视频中）
                frame_hwc = np.clip(frame_hwc, 0, 255).astype(np.uint8)
                bg_img = frame_hwc

            else:
                bg_img = None

            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            time0 = time.time()
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
            print("paste_face: {}".format((time.time() - time0) * 1000))
            tqdm_tool.update(1)

            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None