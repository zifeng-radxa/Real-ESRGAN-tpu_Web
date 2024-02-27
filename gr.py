import os
import uuid
import gradio as gr
from upscale import Upscale

model_list = ['RealESRGAN_x4plus_BF16_rgb_1_3_480_640.bmodel',
              'RealESRGAN_x4plus_anime_6B_rgb_1_3_480_640.bmodel',
              'realesr-animevideo_v3_rgb_1_3_480_640.bmodel',
              'realesr-general-x4v3_dni_0_2_rgb_1_3_480_640.bmodel',
              ]


def run(input_path, model, type, num_worker=1, audio_check=None, face_enhance=None):
    if not os.path.exists(os.path.join('./result', type)):
        os.makedirs(os.path.join('./result', type), exist_ok=True)
    if type == "video":
        output_path = './result/video/out_{}.mp4'.format(uuid.uuid4())
        tmp_path = './result/video/temph264_{}.mp4'.format(uuid.uuid4())
    else:
        output_path = './result/image/out_{}.jpg'.format(uuid.uuid4())
        tmp_path = None
    # try:
    up = Upscale(input_path, output_path, model, tmp_path, type, num_worker=num_worker, face_enhance=face_enhance)
    result_path = up(audio_check)
    # except Exception as e:
    #     print(e)
        # gr.Error("Error, please check the info box")
        # return (e, None)

    return ("Success upscale, click download icon to download to local", result_path)


description = """
# Real-ESRGAN with BM1684M🛸\n
run Real-ESRGAN to upscale video/image resolution by TPU
## Model choose
**RealESRGAN_x4plus**\n
**RealESRGAN_x4plus_anime_6B** optimized for anime images with much smaller model size\n
**realesr-animevideo_v3** optimized for anime videos\n
**realesr-general-x4v3** a tiny small model for general scenes\n
"""

if __name__ == '__main__':
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    up_video = gr.Video(format="mp4", label="upload a video to upscale resolution")
                    with gr.Row():
                        start_button = gr.Button("Start improve", variant="primary")
                        with gr.Column():
                            audio_check = gr.Checkbox(
                                label="audio",
                                info="if click would output with audio"
                            )
                            face_enhance = gr.Checkbox(
                                label="face enhance",
                                info="enhance real world face, coming soon",
                                interactive=False

                            )
                        num_worker = gr.Slider(1, 10, value=4, step=1, label="Thread", info="Choose between 1 and 10", )
                        model = gr.Dropdown(choices=model_list, value=model_list[3], info="select a model",
                                            label="Model",
                                            )
                    info_text = gr.Textbox(label="info output", lines=3)

                    ret_video = gr.Video(label="output video", format=None, autoplay=False)
                    hide_textbox = gr.Textbox(value="video", visible=False)
                    start_button.click(run, [up_video, model, hide_textbox, num_worker, audio_check],
                                       outputs=[info_text, ret_video])

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    up_img = gr.Image(label="upload a image to upscale resolution", type="filepath")
                    with gr.Row():
                        start_button = gr.Button("Start improve", variant="primary")
                        face_enhance_2 = gr.Checkbox(
                            label="face enhance",
                            info="enhance real world face (GFPGANv1Clean run by CPU) it would take time"
                        )
                        audio_check = gr.Checkbox(
                            label="audio",
                            info="if click would output with audio",
                            visible=False
                        )
                        num_worker = gr.Slider(1, 10, value=1, step=1, label="Thread", visible=False, info="Choose between 1 and 10", )

                        model = gr.Dropdown(choices=model_list, value=model_list[0], info="select a model",
                                            label="Model")
                    info_text = gr.Textbox(label="info output", lines=3)

                    ret_img = gr.Image(label="upscale result")
                    hide_textbox = gr.Textbox(value="image", visible=False)
                    start_button.click(run, [up_img, model, hide_textbox, num_worker, audio_check, face_enhance_2], outputs=[info_text, ret_img])

    demo.queue(max_size=2)
    demo.launch(debug=False, show_api=True, share=False, server_name="0.0.0.0")
