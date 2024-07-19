from tools.tpu_utils import load_bmodel
from tools.utils import ratio_resize
import numpy as np
import cv2
class Bgremover():
    def __init__(self):
        self.net = None

    def init_model(self):
        if self.net is None:
            self.net = load_bmodel("rmbg.bmodel")

    def preprocess(self, img):
        img = img.astype('float32')
        img, pad_black = ratio_resize(img, (1024,1024))
        img = img / 255.0
        img[:,:,0] = (img[:,:,0]-0.5)/1.0
        img[:,:,1] = (img[:,:,1]-0.5)/1.0
        img[:,:,2] = (img[:,:,2]-0.5)/1.0
        img = np.expand_dims(img, axis=0)
        img = img.transpose(0, 3, 1, 2)
        return img, pad_black

    def postprocess(self, output):
        output = output.clip(-0.5,0.5)
        output = output * 1 + 0.5
        output = output.transpose(1, 2, 0)
        output = (output * 255.0).round()
        output[output != 255] = 0
        return output

    def forward(self, input):
        img_orl_shape = input.shape[0:2]
        img, pad_black= self.preprocess(input)
        res = self.net([img])[0][0]
        mask = self.postprocess(res)
        img_resize, _ = ratio_resize(input, (1024,1024))
        img_a = cv2.cvtColor(img_resize, cv2.COLOR_RGBA2BGRA)
        img_a[:,:,3] = mask[:,:,0]
        img_a = img_a[:,:,[2,1,0,3]]
        img_a = img_a[pad_black[0]:1024 - pad_black[1], pad_black[2]:1024 - pad_black[3], :]
        img_a = cv2.resize(img_a, img_orl_shape[::-1])
        return img_a, img_a[:,:,3]


