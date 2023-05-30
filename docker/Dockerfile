FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN apt update \
    && apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg g++
WORKDIR /app
RUN git clone https://github.com/snuailab-biz/iwest-fs.git .
RUN python3 -m pip install -r requirements.txt

RUN cd iwestfs/models/ops
RUN python3 setup.py build install
RUN cd ../../..

RUN apt-get update && apt-get install -y \
    libx11-dev \
    libgl1-mesa-glx
ENV DISPLAY=:0


RUN git pull
CMD ["python3", "exe.py"]