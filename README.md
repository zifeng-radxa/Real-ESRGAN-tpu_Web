# Real-ESRGAN-TPU

> Run Real-ESRGAN on BM1684X 

### Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data

> [[Paper](https://arxiv.org/abs/2107.10833)]   [[YouTube Video](https://www.youtube.com/watch?v=fxHWoDSSvSc)]   [[B站讲解](https://www.bilibili.com/video/BV1H34y1m7sS/)]   [[Poster](https://xinntao.github.io/projects/RealESRGAN_src/RealESRGAN_poster.pdf)]   [[PPT slides](https://docs.google.com/presentation/d/1QtW6Iy8rm8rGLsJ0Ldti6kP-7Qyzy6XL/edit?usp=sharing&ouid=109799856763657548160&rtpof=true&sd=true)]
> [Xintao Wang](https://xinntao.github.io/), Liangbin Xie, [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en)
> [Tencent ARC Lab](https://arc.tencent.com/en/ai-demos/imgRestore); Shenzhen Institutes of Advanced Technology, Chinese Academy of Sciences

[![img](https://github.com/xinntao/Real-ESRGAN/raw/master/assets/teaser.jpg)](https://github.com/xinntao/Real-ESRGAN/blob/master/assets/teaser.jpg)



## Usage

- Clone the repo

  ```bash
  git clone https://github.com/zifeng-radxa/Real-ESRGAN-tpu_Web.git
  ```

- setup environment (use sophon-opencv, please setup follow Sophon-mw instruction)

  ```bash
  pip3 install -r requirements.txt
  pip3 install https://github.com/radxa-edge/TPU-Edge-AI/releases/download/v0.1.0/tpu_perf-1.2.31-py3-none-manylinux2014_aarch64.whl
  ```

- run the boot strip 

  ```bash
  bash run.sh
  ```

- use browser access port 7860