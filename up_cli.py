import argparse
from run import run, model_list

parser = argparse.ArgumentParser(description='run IR')
parser.add_argument('--input', '-i', type=str, required=True)
parser.add_argument('--model', '-m', type=str, required=True)
parser.add_argument('--type', '-t', type=str, choices=["video", "image"], required=True)
parser.add_argument('--num_worker', '-n', type=int, default=1)
parser.add_argument('--audio', '-a', action='store_true')
parser.add_argument('--face', '-f', action='store_true')
args = parser.parse_args()

if __name__ == '__main__':
    for model in model_list:
        print(model)
    print("face enhance only support Real world face")
    result = run(input_path=args.input,
                 model=args.model,
                 type=args.type,
                 num_worker=args.num_worker,
                 audio_check=args.audio,
                 face_enhance=args.face)
    print("IR finish!, please check {}".format(result[1]))
