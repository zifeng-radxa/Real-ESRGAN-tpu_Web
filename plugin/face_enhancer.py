import numpy as np
from plugin.face_enhancer_base import FaceEnhancerBase

class CodeFormer(FaceEnhancerBase):
    def __init__(self, code_bmodel, face_bmodel=None, pars_bmodel=None, bg_upsampler=None):
        super().__init__(face_bmodel, pars_bmodel, bg_upsampler)
        self.net = code_bmodel


    def forward(self, img, w):
        img = self.pre_process(img)
        w_np = np.array([w if w is not None else 0.5], dtype=np.float32)
        # t = timeit.default_timer()
        # t = time.time()
        ort_outs = self.net([img, w_np])[0][0]
        output = self.post_process(ort_outs)
        # print('infer time:',timeit.default_timer()-t)
        output = output.astype(np.uint8)
        return output


class GFPGANer(FaceEnhancerBase):
    def __init__(self, gfpgan_bmodel, face_bmodel=None, pars_bmodel=None, bg_upsampler=None):
        super().__init__(face_bmodel, pars_bmodel, bg_upsampler)
        self.net = gfpgan_bmodel

    def forward(self, img, w=None):
        img = self.pre_process(img)
        # t = timeit.default_timer()
        # t = time.time()
        ort_outs = self.net([img])[0][0]
        output = self.post_process(ort_outs)
        # print('infer time:',timeit.default_timer()-t)
        output = output.astype(np.uint8)
        return output
