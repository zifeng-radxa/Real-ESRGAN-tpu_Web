import gradio as gr
from core.pipeline import image_pipeline, image_model_list, video_pipeline, video_model_list
from tools.utils import get_host_ip
description = """
# Real-ESRGAN with Airbox🛸\n
run Real-ESRGAN to upscale video/image resolution by TPU
## Model choose
**RealESRGAN_x4plus**\n
**RealESRGAN_x4plus_anime_6B** optimized for anime images with much smaller model size\n
**realesr-animevideo_v3** optimized for anime videos\n
**realesr-general-x4v3** a tiny small model for general scenes\n
"""
# def change_thread_num(background_remove_v):
#     print(background_remove_v)
#     if background_remove_v == 0:
#         return gr.Slider(1, 4, value=1, step=1, label="Thread", info='upscale + remove background only accept 1 thread' )
#     else:
#         return gr.Slider(1, 4, value=1, step=1, label="Thread", info="Choose between 1 and 2", )

if __name__ == '__main__':
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown(description)
        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column():
                    up_video = gr.Video(format="mp4", label="upload a video to upscale resolution")
                    model = gr.Dropdown(choices=video_model_list, value=video_model_list[0], info="select a model",
                                        label="Model",
                                        )
                    with gr.Row():
                        face_enhance_v = gr.Radio(
                            choices=["GFPGAN", "CodeFormer", "None"],
                            value="None",
                            type="value",
                            label="Face Enhance Tools",
                        )

                        background_remove_v = gr.Radio(
                            choices=["upscale + remove background",
                                     "only remove background",
                                     "None"],
                            value="None",
                            type="index",
                            label="Background Remove tool"
                        )
                    with gr.Row():
                        audio = gr.Checkbox(
                            label="audio",
                            info="if click would output with audio",
                            value=True
                        )
                        num_worker = gr.Slider(1, 4, value=2, step=1, label="Thread", info="Choose between 1 and 4", )

                    with gr.Row():
                        clear_button_v = gr.ClearButton(value="Clear", components=[up_video])
                        start_button_v = gr.Button("Start improve", variant="primary", scale=1)

                with gr.Column():
                    ret_video = gr.Video(label="output video", format=None, autoplay=False)
                    info_text = gr.Textbox(label="info output", lines=3)

        # background_remove_v.change(change_thread_num, background_remove_v, num_worker)
        clear_button_v.add(components=[info_text, ret_video])
        start_button_v.click(video_pipeline, [up_video, model, face_enhance_v, background_remove_v, num_worker, audio],
                           outputs=[info_text, ret_video])

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    up_img = gr.Image(label="upload a image to upscale resolution", type="numpy", scale=1)
                    model = gr.Dropdown(choices=image_model_list, value=image_model_list[0], info="select a model",
                                        label="Model", scale=1)

                    face_enhance_i = gr.Radio(
                        choices=["GFPGAN", "CodeFormer", "None"],
                        value="None",
                        type="value",
                        label="Face Enhance Tools",
                    )

                    with gr.Row():
                        background_remove_i = gr.Radio(
                            choices=["upscale + remove background",
                                     "only remove background",
                                     "None"],
                            value="None",
                            type="index",
                            label="Background Remove tool"
                        )
                        id_bg_color = gr.Radio(
                            choices=["White",
                                     "Blue",
                                     "Red",
                                     "Transparent",
                                     "Pick color"],
                            value="White",
                            type="value",
                            label="ID photo background photo"
                        )

                        user_bg_color = gr.ColorPicker(label="pick your background color", value="#ffffff")

                    with gr.Row():
                        clear_button = gr.ClearButton(value="Clear")
                        start_button = gr.Button("Start improve", variant="primary",scale=1)

                    with gr.Accordion(label="Advanced", open=False):
                        bg_img = gr.Image(label="upload your background picture")
                        bg_img_up = gr.Checkbox(
                            label="up scale background picture",
                            info="if click would up scale background picture",
                            value=False
                        )


                with gr.Column():
                    ret_img = gr.Image(label="upscale result", format="png", scale=1)
                    info_text = gr.Textbox(label="info output", lines=3)

            clear_button.add(components=[up_img, info_text, ret_img, bg_img, bg_img_up])
            start_button.click(image_pipeline, [up_img, model, face_enhance_i, background_remove_i, id_bg_color, user_bg_color, bg_img, bg_img_up], outputs=[info_text, ret_img])

    demo.queue(max_size=10)
    demo.launch(debug=False, show_api=True, share=False, server_name=get_host_ip())
