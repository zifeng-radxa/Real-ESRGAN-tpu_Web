from FACEXLIB.facexlib.utils.face_restoration_helper import FaceRestoreHelper
import cv2
import numpy as np
import time

class FaceEnhanceBase():
    def __init__(self, face_bmodel=None, pars_bmodel=None, bg_upsampler=None):
        self.net = None
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

    def pre_process(self, img):
        img = img / 255.0
        img = img.astype('float32')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img[:,:,0] = (img[:,:,0]-0.5)/0.5
        img[:,:,1] = (img[:,:,1]-0.5)/0.5
        img[:,:,2] = (img[:,:,2]-0.5)/0.5
        img = np.float32(img[np.newaxis,:,:,:])
        img = img.transpose(0, 3, 1, 2)
        return img

    def post_process(self, output):
        output = output.clip(-1,1)
        output = (output + 1) / 2
        output = output.transpose(1, 2, 0)
        output = (output * 255.0).round()

        return output

    def forward(self, img, w):
        pass

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
            restored_face = self.forward(cropped_face, w)
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
