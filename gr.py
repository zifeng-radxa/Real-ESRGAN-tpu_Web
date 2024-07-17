import gradio as gr
from core.pipeline import image_pipeline, model_list
from core.imageupscaler import ImageUpscaler

description = """
# Real-ESRGAN with Airbox🛸\n
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
        # with gr.Tab("Video"):
        #     with gr.Row():
        #         with gr.Column():
        #             up_video = gr.Video(format="mp4", label="upload a video to upscale resolution")
        #             with gr.Row():
        #                 start_button = gr.Button("Start improve", variant="primary")
        #                 with gr.Column():
        #                     audio_check = gr.Checkbox(
        #                         label="audio",
        #                         info="if click would output with audio"
        #                     )
        #                     face_enhance = gr.Checkbox(
        #                         label="face enhance",
        #                         info="enhance real world face",
        #                         interactive=True
        #
        #                     )
        #                 num_worker = gr.Slider(1, 10, value=4, step=1, label="Thread", info="Choose between 1 and 10", )
        #                 model = gr.Dropdown(choices=model_list, value=model_list[3], info="select a model",
        #                                     label="Model",
        #                                     )
        #             info_text = gr.Textbox(label="info output", lines=3)
        #
        #             ret_video = gr.Video(label="output video", format=None, autoplay=False)
        #             hide_textbox = gr.Textbox(value="video", visible=False)
        #             start_button.click(run, [up_video, model, hide_textbox, num_worker, audio_check, face_enhance],
        #                                outputs=[info_text, ret_video])

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    up_img = gr.Image(label="upload a image to upscale resolution", type="numpy")
                    with gr.Row():
                        model = gr.Dropdown(choices=model_list, value=model_list[0], info="select a model",
                                            label="Model", scale=1)
                        with gr.Column():
                            face_enhance_2 = gr.Radio(
                                choices=["GFPGAN", "CodeFormer", "None"],
                                value="None",
                                type="value",
                                label="Face Enhance Tools",
                            )

                            background_remove = gr.Checkbox(
                                label="remove background",
                            )

                        start_button = gr.Button("Start improve", variant="primary",scale=1)

                    info_text = gr.Textbox(label="info output", lines=3)

                    ret_img = gr.Image(label="upscale result")
                    hide_textbox = gr.Textbox(value="image", visible=False)
                    start_button.click(image_pipeline, [up_img, model, face_enhance_2, background_remove], outputs=[info_text, ret_img])

    demo.queue(max_size=10)
    demo.launch(debug=False, show_api=True, share=False, server_name="0.0.0.0")
