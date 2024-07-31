import os
from tools.tpu_utils import load_bmodel
from tools.utils import timer
import numpy as np
import cv2
from threading import Thread
from tqdm import tqdm
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor


class Bgremover():
    def __init__(self):
        self.net = None
        self.cur_name = None

    def init_model(self, model_name="rmbg_f16_1.bmodel"):
        if model_name != self.cur_name:
            self.net = load_bmodel(model_name)
            self.cur_name = model_name

    def clean_tpu_memory(self):
        self.net = None
        self.cur_name = None

    def preprocess(self, img):
        img = img.astype('float32')
        # img, pad_black = ratio_resize(img, (1024,1024))
        img = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        img = img / 255.0
        img[:, :, 0] = (img[:, :, 0] - 0.5) / 1.0
        img[:, :, 1] = (img[:, :, 1] - 0.5) / 1.0
        img[:, :, 2] = (img[:, :, 2] - 0.5) / 1.0
        img = np.expand_dims(img, axis=0)
        img = img.transpose(0, 3, 1, 2)
        return img, None

    def postprocess(self, output, img_orl_size, only_black_white=False):
        ma = np.max(output)
        mi = np.min(output)
        output = (output - mi) / (ma - mi)
        # output = output.clip(-0.5,0.5)
        # output = output * 1 + 0.5
        output = output.transpose(1, 2, 0)
        output = (output * 255.0).round()
        output = cv2.resize(output, (img_orl_size[1], img_orl_size[0]), interpolation=cv2.INTER_LINEAR)
        if only_black_white:
            output[output > 250] = 255
            output[output <= 250] = 0
        # output = np.expand_dims(output, axis=-1)
        output = output.astype(np.uint8)
        return output

    def forward(self, input, only_black_white=False):
        img_orl_shape = input.shape[0:2]
        img, pad_black = self.preprocess(input)
        res = self.net([img])[0][0]
        mask = self.postprocess(res, img_orl_shape, only_black_white)
        img_a = cv2.cvtColor(input, cv2.COLOR_RGB2RGBA)
        img_a[:, :, 3] = mask
        # img_a = img_a[:, :, [2, 1, 0, 3]]
        img_a = cv2.resize(img_a, img_orl_shape[::-1])
        return img_a, img_a[:, :, 3]


class Bgremover2(Bgremover):
    def __init__(self, thread_num=1):
        super().__init__()
        self.nets = []
        self.thread_num = thread_num
        self.video_shape = None

    def init_model(self, thread_num=1):
        self.thread_num = thread_num
        cur_net = len(self.nets)
        diff = 1 - cur_net
        if diff > 0:
            for i in range(diff):
                net = load_bmodel('rmbg_f16_30.bmodel', model_type="video")
                self.nets.append(net)
        else:
            for i in range(-diff):
                self.nets.pop()

    def clean_tpu_memory(self):
        self.nets.clear()
        self.thread_num = 1
        self.video_shape = None


    def postprocess(self, output, img_orl_size, keep_shape=True):
        ma = np.max(output)
        mi = np.min(output)
        output = (output - mi) / (ma - mi)
        # output = output.clip(-0.5,0.5)
        # output = output * 1 + 0.5
        output = output.transpose(1, 2, 0)
        output = (output * 255.0).round()
        output = cv2.resize(output, (img_orl_size[1], img_orl_size[0]), interpolation=cv2.INTER_LINEAR)
        output[output > 250] = 255
        output[output <= 250] = 0
        # output = np.expand_dims(output, axis=-1)
        output = output.astype(np.uint8)
        return output

    async def async_imwrite(self, path, img):
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, cv2.imwrite, path, img)

    async def thread_progress(self, input_file_name, worker, source_path='./temp_frames',
                              target_path='./temp_res_frames', save_type="image", keep_shape=True):
        masks_data = []
        imgs_data = [cv2.imread(os.path.join(source_path, i)) for i in input_file_name]
        imgs_infer_data = [self.preprocess(i)[0] for i in imgs_data]

        input_numpy = np.concatenate(imgs_infer_data, axis=0)
        imgs_infer_data.clear()
        res_numpy = worker([input_numpy])[0]

        async_mask_task = []
        for i in range(len(res_numpy)):
            if save_type == "mask":
                mask = self.postprocess(res_numpy[i], (1440, 2560))
                task = self.async_imwrite(os.path.join('./temp_mask', input_file_name[i]), mask)
                async_mask_task.append(task)
            else:
                mask = self.postprocess(res_numpy[i], self.video_shape)
                masks_data.append(mask)
        del res_numpy
        if len(async_mask_task) != 0 :
            await asyncio.gather(*async_mask_task)

        if save_type == "image":
            async_task = []
            for i in range(len(input_file_name) - 1, -1, -1):
                img_orl = imgs_data.pop()
                mask = masks_data.pop()
                img_orl = img_orl.astype(np.float32)
                green_img = np.zeros_like(img_orl)
                green_img[:, :] = [0, 255, 0]
                green_area = cv2.bitwise_and(green_img, green_img, mask=cv2.bitwise_not(mask))
                res = cv2.bitwise_and(img_orl, img_orl, mask=mask)
                res = cv2.add(green_area, res)
                # await self.async_imwrite(os.path.join(target_path, input_file_name[i]), res)
                task = self.async_imwrite(os.path.join(target_path, input_file_name[i]), res)
                async_task.append(task)

            await asyncio.gather(*async_task)


    def run_thread_progress(self, input_file_name, worker, source_path, save_type, tqdm_tool=None):
        asyncio.run(self.thread_progress(input_file_name, worker, source_path, save_type=save_type))
        tqdm_tool.update(1)

    async def io_bgrm_thread(self, input_file_name):
        mask_data = [cv2.imread(os.path.join('temp_mask', i), cv2.IMREAD_GRAYSCALE) for i in input_file_name]
        up_img_data = [cv2.imread(os.path.join('temp_res_frames', i)) for i in input_file_name]
        async_task = []
        for i in range(len(mask_data) - 1, -1, -1):
            mask = mask_data.pop()
            img = up_img_data.pop()
            mask = mask.astype(np.uint8)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            img = img.astype(np.float32)
            green_img = np.zeros_like(img)
            green_img[:, :] = [0, 255, 0]
            green_area = cv2.bitwise_and(green_img, green_img, mask=cv2.bitwise_not(mask))
            cv2.imwrite(os.path.join('temp_test', input_file_name[i]), green_area)
            res = cv2.bitwise_and(img, img, mask=mask)
            res = cv2.add(green_area, res)
            task = self.async_imwrite(os.path.join('temp_res_frames', input_file_name[i]), res)
            async_task.append(task)

        await asyncio.gather(*async_task)

    def run_io_bgrm_progress(self, input_file_name, tqdm_tool=None):
        # print(input_file_name)
        time0 = time.time()
        asyncio.run(self.io_bgrm_thread(input_file_name))
        # print((time.time() - time0) * 1000)
        tqdm_tool.update(1)

    def get_video_meta(self, input):
        img = cv2.imread(os.path.join('./temp_frames', input))
        self.video_shape = img.shape[:2]

    def forward(self, input, save_type="image", call_back="run_thread_progress"):
        if isinstance(input, str):
            imgs_file = os.listdir(input)
            total_img_num = len(imgs_file)
            self.get_video_meta(imgs_file[0])
            SPLIT = 30
            task_num = total_img_num // SPLIT if total_img_num % SPLIT == 0 else total_img_num // SPLIT + 1
            frame_split = [SPLIT for _ in range(task_num - 1)]
            frame_split.append(SPLIT if total_img_num % SPLIT == 0 else total_img_num % SPLIT)
            print(frame_split)
            tqdm_tool = tqdm(total=task_num)
            with ThreadPoolExecutor(self.thread_num) as t:  # 更改线程池数量
                for i in range(0, task_num):
                    sub_img_file = imgs_file[i * SPLIT:(i * SPLIT + SPLIT)]
                    if call_back == "run_thread_progress":
                        t.submit(self.run_thread_progress,
                                 sub_img_file, self.nets[0], input, save_type, tqdm_tool)
                        # thread.append(t)
                    elif call_back == "run_io_bgrm_progress":
                        t.submit(self.run_io_bgrm_progress, sub_img_file,tqdm_tool)

