ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:20.02-py3
FROM $BASE_IMAGE

RUN pip install matplotlib==3.3.4
RUN pip install opencv-python==4.4.0.46
RUN pip install opt-einsum==3.2.1
RUN pip install pandas==0.25.0
RUN pip install pytorch-lightning==1.2.3
RUN pip install tensorboardX==2.1
RUN pip install torch==1.8.0
RUN pip install torchvision==0.9.0
RUN pip install tqdm==4.59.0
RUN pip install webdataset @ git+https://github.com/tmbdev/webdataset.git@a31cd1418c4633f69772c8591a9db39cbd571ab2