import os
import uuid
import gradio as gr
from upscale import Upscale, EngineOV



# model_odd = load_model()
# model_even = load_model()

def run(input_path, audio_check, num_worker):
    if not os.path.exists('./result'):
        os.mkdir('./result')
    output_path = './result/out_{}.mp4'.format(uuid.uuid4())
    tmp_path = './result/temph264_{}.mp4'.format(uuid.uuid4())
    # try:
    up = Upscale(input_path, output_path, tmp_path, num_worker=num_worker)
    result_path = up(audio_check)
    # except Exception as e:
    #     return (e, None)

    return ("Success upscale video, click download icon to download to local", result_path)


description = """
# Real-ESRGAN with BM1684M🛸\n
run Real-ESRGAN to upscale video resolution by TPU
"""

if __name__ == '__main__':
    with gr.Blocks(analytics_enabled=False) as demo:
        with gr.Row():
            gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                up_video = gr.Video(format="mp4", label="upload a video to upscale resolution")
                with gr.Row():
                    start_button = gr.Button("Start improve")
                    audio_check = gr.Checkbox(
                        label="audio",
                        info="if click would output with audio"
                    )
                    num_worker = gr.Dropdown(choices=[1, 2, 3, 4, 5, 6], value=6, info="num of workers", label="Threads")

                info_text = gr.Textbox(label="info output", lines=3)

                ret_video = gr.Video(label="output video", format=None, autoplay=False)
                start_button.click(run, [up_video, audio_check, num_worker], outputs=[info_text, ret_video])

    demo.queue(max_size=2)
    demo.launch(debug=True, show_api=True, share=False, server_name="0.0.0.0")


