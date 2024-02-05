import streamlit as st
# from upscale import EngineOV


MODEL_PATH = './model/realesrgan-x4_BF16_480.bmodel'
DEVICE_ID = 0



# @st.cache_resource
# def load_model():
#     model = EngineOV(model_path=MODEL_PATH, batch=1, device_id=DEVICE_ID)
#     print('BM_MODEL LOADED')
#     return model


description = """
# Real-ESRGAN with BM1684M🛸\n
run Real-ESRGAN to upscale video resolution by TPU
"""


if __name__ == '__main__':
    st.markdown(description)

# video_file = open('./result/temp_9ac3dc05-a38d-47d9-a080-df40f4e37f5d.avi', 'rb')
# video_bytes = video_file.read()
#
# st.video(video_bytes)