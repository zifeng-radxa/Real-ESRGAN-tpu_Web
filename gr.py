import os
import uuid
import gradio as gr
from upscale import Upscale

model_list = ['realesrgan-x4_BF16_480.bmodel']


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
        up = Upscale(input_path, output_path, model, tmp_path, type, num_worker=num_worker)
        result_path = up(audio_check)
    except Exception as e:
        gr.Error("Error, please check the info box")
        return (e, None)

    return ("Success upscale, click download icon to download to local", result_path)


description = """
# Real-ESRGAN with BM1684M🛸\n
run Real-ESRGAN to upscale video/image resolution by TPU
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
                        start_button = gr.Button("Start improve")
                        audio_check = gr.Checkbox(
                            label="audio",
                            info="if click would output with audio"
                        )
                        num_worker = gr.Slider(1, 10, value=4, step=1, label="Thread", info="Choose between 1 and 10")
                        model = gr.Dropdown(choices=model_list, value=model_list[0], info="select a model",
                                            label="Model")
                    info_text = gr.Textbox(label="info output", lines=3)

                    ret_video = gr.Video(label="output video", format=None, autoplay=False)
                    hide_textbox = gr.Textbox(value="video", visible=False)
                    start_button.click(run, [up_video, model, hide_textbox, num_worker, audio_check], outputs=[info_text, ret_video])

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    up_img = gr.Image(label="upload a image to upscale resolution", type="filepath")
                    with gr.Row():
                        start_button = gr.Button("Start improve")
                        model = gr.Dropdown(choices=model_list, value=model_list[0], info="select a model",
                                            label="Model")
                    info_text = gr.Textbox(label="info output", lines=3)

                    ret_img = gr.Image(label="upscale result")
                    hide_textbox = gr.Textbox(value="image", visible=False)
                    start_button.click(run, [up_img, model, hide_textbox], outputs=[info_text, ret_img])


    demo.queue(max_size=2)
    demo.launch(debug=False, show_api=True, share=False, server_name="0.0.0.0")
