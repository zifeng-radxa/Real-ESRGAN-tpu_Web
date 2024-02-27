import os
import ffmpeg

class Writer:
    def __init__(self, audio, ouput_path, temp_path, fps):
        os.environ["LD_LIBRARY_PATH"] = "/opt/sophon/libsophon-current/lib:"
        out_width, out_height = 2560, 1920
        if audio:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 audio,
                                 ouput_path,
                                 pix_fmt='yuv420p',
                                 vcodec='libx264',
                                 loglevel='error',
                                 acodec='copy').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))
        else:
            self.stream_writer = (
                ffmpeg.input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{out_width}x{out_height}',
                             framerate=fps).output(
                                 temp_path, pix_fmt='yuv420p', vcodec='h264',
                                 loglevel='error').overwrite_output().run_async(
                                     pipe_stdin=True, pipe_stdout=True, cmd='ffmpeg'))

    def write_frame(self, frame):
        os.environ["LD_LIBRARY_PATH"] = "/opt/sophon/libsophon-current/lib:"
        frame = frame.tobytes()
        self.stream_writer.stdin.write(frame)

    def close(self):
        self.stream_writer.stdin.close()
        self.stream_writer.wait()

