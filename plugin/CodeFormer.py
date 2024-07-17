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
        # self.face_helper = None
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

    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, w=0.5, tqdm_tool=None, bg_upscale_img=None):
        self.face_helper.clean_all()
        self.face_helper.read_image(img)
        time0 = time.time()
        self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
        self.face_helper.align_warp_face()

        if tqdm_tool is not None:
            tqdm_tool.update(1)
        # print("face detect: {}".format((time.time() - time0) * 1000))

        for cropped_face in self.face_helper.cropped_faces:
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0)  # .to(devices.device_codeformer)
            try:
                # in: (1, 3, 512, 512)    out: (1, 3, 512, 512) (1, 256, 1024) (1, 256, 16, 16)
                output = self.net([cropped_face_t.numpy(), np.array([w if w is not None else 0.5], dtype=np.float32)])[
                        0]  ## the dtype must be explicitly set
                restored_face = tensor2img(torch.from_numpy(output).squeeze(0), rgb2bgr=False, min_max=(-1, 1))
            # del output
                # print("face enhance: {}".format((time.time() - time0) * 1000))

            except Exception:
                print('Failed inference for CodeFormer')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        if tqdm_tool is not None:
            tqdm_tool.update(1)

        if not has_aligned and paste_back:
            self.face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_upscale_img)
            tqdm_tool.update(1)
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, restored_img
        else:
            return self.face_helper.cropped_faces, self.face_helper.restored_faces, None