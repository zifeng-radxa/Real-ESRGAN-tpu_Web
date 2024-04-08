# from GFPGAN.gfpgan.utils import GFPGANer
from CodeFormer import FaceRestorerCodeFormer

class FaceEnhance():
    def __init__(self, upsampler, face_bmodel, pars_bmodel, enhance_bmodel):
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

        self.face_enhancer = FaceRestorerCodeFormer(
            code_bmodel= enhance_bmodel,
            face_bmodel= face_bmodel,
            pars_bmodel = pars_bmodel,
            bg_upsampler= upsampler

        )

    def run(self, img, inference_frames, tqdm_tool=None):
        if isinstance(img, list):
            for _ in range(len(img)):
                _img = img.pop()
                _, _, output = self.face_enhancer.enhance(_img['data'], has_aligned=False, only_center_face=False,
                                                          paste_back=True, tqdm_tool=None)
                inference_frames.append({"id": _img["id"], "data": output})
                tqdm_tool.update(1)

        else:
            _, _, output = self.face_enhancer.enhance(img['data'], has_aligned=False, only_center_face=False, paste_back=True, tqdm_tool=tqdm_tool)
            inference_frames.append({"id": img["id"], "data": output})
        # return output