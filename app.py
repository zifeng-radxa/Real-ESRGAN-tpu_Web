import base64
import os
import uuid
import streamlit as st
#from streamlit_image_select import image_select
#from st_clickable_images import clickable_images
from upscale import Upscale
from streamlit_image_comparison import image_comparison

#from PIL import Image
MODEL_PATH = './model/realesrgan-x4_BF16_480.bmodel'
DEVICE_ID = 0



# @st.cache_resource
# def load_model():
#     model = EngineOV(model_path=MODEL_PATH, batch=1, device_id=DEVICE_ID)
#     print('BM_MODEL LOADED')
#     return model
def run(input_path, model, type, num_worker=1, audio_check=None):
    if not os.path.exists(os.path.join('./result', type)):
        os.makedirs(os.path.join('./result', type), exist_ok=True)
    if type == "video":
        output_path = './result/video/out_{}.mp4'.format(uuid.uuid4())
        tmp_path = './result/video/temph264_{}.mp4'.format(uuid.uuid4())
    else:
        output_path = './result/image/out_{}.jpg'.format(uuid.uuid4())
        tmp_path = None
    try:
        with st.spinner('Please Wait'):
            up = Upscale(input_path, output_path, model, tmp_path, type, num_worker=num_worker)
            result_path = up(audio_check)
        st.write("Upscale result")
        if type == 'image':
            # st.image(result_path)
            pass
        else:
            st.video(result_path)
        st.snow()

        return result_path

    except Exception as e:
        st.error("Error, please check error information")
        st.write(e)
        # return (e, None)



def download_bytes(result_path, type):
    if type == "image":
        file_name = "upscale_result.jpg"
    else:
        file_name = "upscale_result.mp4"
    with open(result_path, "rb") as f:
        file_bytes = f.read()

    st.download_button(
        label="Download",
        data=file_bytes,
        file_name=file_name
    )


    # return ("Success upscale, click download icon to download to local", result_path)

description = """
# Real-ESRGAN with BM1684M🛸\n
run Real-ESRGAN to upscale video/image resolution by TPU
"""


if __name__ == '__main__':
    st.markdown(description)
    # with st.sidebar:
    #     st.markdown("""
    #     # Real-ESRGAN
    #     """)
    #     model_name = st.selectbox(label="Model_t", options=['realesrgan-x4_BF16_480.bmodel'], index=0)

    # video, image, test = st.tabs(["video", "image", "test"])
    video, image = st.tabs(["video", "image"])
    with video:
        up_video = st.file_uploader(label="Upload file", type=['mp4', 'avi'])
        if up_video is not None:
            video_bytes = up_video.getvalue()
            st.video(video_bytes)
            # button = st.button(label="Start Upscale")
        with st.container():
            button_c, adv_c= st.columns([1.5, 5])
            with button_c:
                button = st.button(label="Start Upscale", key="video_button", type="primary")
            with adv_c:
                with st.expander("advanced"):
                    thread_c, model_c = st.columns([1, 2])
                    with thread_c:
                        thread = st.slider(label="Thread", help="Choose between 1 and 10", min_value=1, max_value=10,
                                           value=4,
                                           step=1)
                    with model_c:
                        model_name = st.selectbox(label="Model", options=['realesrgan-x4_BF16_480.bmodel'], index=0,
                                                  key="video_model")

            audio = st.checkbox(label="Audio output", help="output video include audio")
        if button:
            try:
                cache_name = 'video_{}.mp4'.format(uuid.uuid4())
                cache_path = './cache/{}'.format(cache_name)
                if not os.path.exists('./cache'):
                    os.makedirs('./cache', exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(video_bytes)
            except Exception as e:
                st.error("Please upload file")
            result_path = run(cache_path, model_name, "video", thread, audio)
            os.remove(cache_path)
            download_bytes(result_path, "video")


    with image:
        up_image = st.file_uploader(label="Upload file", type=['jpeg', 'jpg', 'png'], accept_multiple_files=False)
        if up_image is not None:
            image_bytes = up_image.getvalue()
            st.image(image_bytes, caption="original image", width=224)

        with st.container():
            button_c, adv_c= st.columns([1.5, 5])
            with button_c:
                button = st.button(label="Start Upscale", key="image_button", type="primary")

            with adv_c:
                with st.expander("advanced"):
                    model_name = st.selectbox(label="Model", options=['realesrgan-x4_BF16_480.bmodel'], index=0, key="image_model")

        if button:
            try:
                cache_name = 'image_{}.jpg'.format(uuid.uuid4())
                cache_path = './cache/{}'.format(cache_name)
                if not os.path.exists('./cache'):
                    os.makedirs('./cache', exist_ok=True)
                with open(cache_path, "wb") as f:
                    f.write(image_bytes)
            except Exception as e:
                st.error('Please upload file')

            result_path = run(cache_path, model_name, "image")
            # st.write("Upscale Comparison")
            image_comparison(
                img1=cache_path,
                img2=result_path,
                label1="original",
                label2="upscale",
                in_memory=True
            )
            os.remove(cache_path)
            download_bytes(result_path, "image")


        # with st.image()
    # with test:
    #     img = image_select(
    #         label="select a cat",
    #         images=['./result/aaa.jpg' for i in range(8)],
    #         captions=['aaaaaaa' for _ in range(8)],
    #         # use_container_width=False
    #         return_value='index',
    #         index=-1
    #     )
    #     st.write(img)
    #     images = []
    #     for file in ["./result/aaa.jpg" for _ in range(10)]:
    #         with open(file, "rb") as image:
    #             encoded = base64.b64encode(image.read()).decode()
    #             images.append(f"data:image/jpeg;base64,{encoded}")
    #
    #     clicked = clickable_images(
    #         images,
    #         titles=[f"Image #{str(i)}" for i in range(len(images))],
    #         div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    #         img_style={"margin": "20px", "height": "300px"},
    #     )
    #     # images = []
    #     # f = open('./result/aaa.jpg', 'rb')
    #     # encoded = base64.b64encode(f.read()).decode()
    #     # f.close()
    #     #     # images.append(encoded)
    #     # clicked = clickable_images(
    #     #     ['https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=700',
    #     #      'https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=700',
    #     #      'https://images.unsplash.com/photo-1565130838609-c3a86655db61?w=700',
    #     #      encoded],
    #     #     div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    #     #     img_style={"margin": "10px", "height": "200px"},
    #     #
    #     # )
    #     st.markdown(f"Image #{clicked} clicked" if clicked > -1 else "No image clicked")
    #     st.page_link("./pages/son_page_1.py", label="Page 2", icon="2️⃣")
    #
    #     st.columns([1,1,1,1])
    #     # img = cv2.imread('./result/aaa.jpg')
    #     # button_img = st.button(st.image(img, channels="BGR"))
    #
    #     for i in range(20):
    #         # c_1, c_2, c_3, c_4 = st.columns([1,1,1,1])
    #         for i in st.columns([1,1,1,1]):
    #             with i:
    #                 with st.container():
    #                     st.image('./result/aaa.jpg', width=160)
    #                     st.page_link("./pages/son_page_1.py", label="zzf")







# video_file = open('./result/temp_9ac3dc05-a38d-47d9-a080-df40f4e37f5d.avi', 'rb')
# video_bytes = video_file.read()
#
# st.video(video_bytes)