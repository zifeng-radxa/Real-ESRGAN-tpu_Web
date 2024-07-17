# from GFPGAN.gfpgan.utils import GFPGANer
from plugin.CodeFormer import FaceRestorerCodeFormer
from plugin.gfpganer import GFPGANer
class FaceEnhance():
    def __init__(self, re_upscale_model, face_detect_model, face_pars_model, face_enhance_model, name):
        # self.face_enhancer = GFPGANer(
        #     model_path='./model/GFPGANv1.3.pth',
        #     upscale=4,
        #     arch='clean',
        #     channel_multiplier=2,
        #     bg_upsampler=upsampler,
        #     face_bmodel = face_bmodel,
        #     pars_bmodel = pars_bmodel,
        #     gfgan_bmodel = enhance_bmodel
        # )
        self.enhance_name = name
        self.face_detect_model = face_detect_model
        self.face_pars_model = face_pars_model
        self.re_upscale_model = re_upscale_model
        self.face_enhance_model = face_enhance_model
        self.face_enhancer = None
        self.init_enhancer()

    def init_enhancer(self):
        if self.enhance_name == "CodeFormer":
            self.face_enhancer = FaceRestorerCodeFormer(
                code_bmodel=self.face_enhance_model,
                face_bmodel=self.face_detect_model,
                pars_bmodel=self.face_pars_model,
                bg_upsampler=self.re_upscale_model
            )

        elif self.enhance_name == "GFPGAN":
            self.face_enhancer = GFPGANer(
                gfpgan_bmodel=self.face_enhance_model,
                face_bmodel=self.face_detect_model,
                pars_bmodel=self.face_pars_model,
                bg_upsampler=self.re_upscale_model
            )

    def run(self, img, tqdm_tool=None, bg_upscale_img=None):
        # if isinstance(img, list):
        #     for _ in range(len(img)):
        #         _img = img.pop()
        #         _, _, output = self.face_enhancer.enhance(_img['data'], has_aligned=False, only_center_face=False,
        #                                                   paste_back=True, tqdm_tool=None)
        #         inference_frames.append({"id": _img["id"], "data": output})
        #         tqdm_tool.update(1)

        # else:
        _, _, output = self.face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True, tqdm_tool=tqdm_tool, bg_upscale_img=bg_upscale_img)
        # inference_frames.append({"id": img["id"], "data": output})
        return output