import os
from tpu_perf.infer import SGInfer
import time
MODEL_PATH = './model/'
DEVICE_ID = 0



def load_model(model_name):
    model = EngineOV(model_path=os.path.join(MODEL_PATH, model_name), batch=1, device_id=DEVICE_ID)
    return model

class EngineOV:
    def __init__(self, model_path="", batch=1 ,device_id=0) :
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            print(">>>> device_id is in os.environ. and device_id = " ,device_id)
        self.model = SGInfer(model_path , batch=batch, devices=[device_id])

    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path ,self.device_id)

    def __call__(self, args):
        start = time.time()
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
            # print(values)
        task_id = self.model.put(*values)
        task_id, results, valid = self.model.get()
        # print(str(round((time.time() - start) * 1000, 3)) + " ms")
        return results