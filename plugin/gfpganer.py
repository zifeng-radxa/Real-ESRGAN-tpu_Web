import torch
import cv2
import numpy as np
from FACEXLIB.facexlib.utils.face_restoration_helper import FaceRestoreHelper
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from plugin.gfpganruntime import GFPGANFaceAugment
import time

class GFPGANer(GFPGANFaceAugment):
    def __init__(self, gfpgan_bmodel=None, face_bmodel=None, pars_bmodel=None, bg_upsampler=None):
        super().__init__(gfpgan_bmodel=gfpgan_bmodel)

        # self.net = gfpgan_bmodel
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

    def enhance(self, img, has_aligned=False, only_center_face=False, paste_back=True, weight=0.5, tqdm_tool=None, bg_upscale_img=None):
        # _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        self.face_helper.clean_all()
        if has_aligned:  # the inputs are already aligned
            img = cv2.resize(img, (512, 512))
            self.face_helper.cropped_faces = [img]
        else:
            self.face_helper.read_image(img)
            self.face_helper.get_face_landmarks_5(only_center_face=only_center_face, eye_dist_threshold=5)
            self.face_helper.align_warp_face()
        if tqdm_tool is not None:
            tqdm_tool.update(1)

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            restored_face, _ = self.forward(cropped_face)
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

