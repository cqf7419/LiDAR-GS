
## 如果笔记本是M芯片 加一个--platform
FROM --platform=linux/amd64 ubuntu:20.04
# 避免交互对话过程
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && apt-get update && apt-get install -y gnupg

RUN apt-get clean 
RUN rm -r /var/lib/apt/lists/* 
RUN apt-key update 
RUN apt-get update 
RUN apt-get update && apt-get install -y wget
RUN apt-get install -y unzip
RUN apt-get install -y lsof
RUN apt-get install -y python3-pip
RUN apt-get install -y curl
RUN apt-get install -y jq
RUN apt-get install -y vim
RUN apt-get install -y net-tools
# RUN apt-get install -y --allow-unauthenticated ssh


# Use the above args
# Install miniconda to /miniconda  
RUN wget "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda
RUN conda init


# 注意使用root来启动训练任务，区分大小写
USER root

RUN apt-get install sudo
RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple
# 安装jupyterlab等python包
RUN pip install jupyterlab yacs \
    && chmod 777 /root \
    && ln -s `which jupyter` /usr/local/bin/jupyter || true \
    && apt-get install -y --allow-unauthenticated iputils-ping

# 安装cuda
RUN mkdir install
COPY ./cuda_12.2.0_535.54.03_linux.run ./install/
RUN sudo chmod +x ./install/cuda_12.2.0_535.54.03_linux.run
RUN apt-get install -y libxml2
RUN sudo ./install/cuda_12.2.0_535.54.03_linux.run --toolkit --samples --silent
RUN echo 'export PATH=/usr/local/cuda/bin:$PATH' | sudo tee /etc/profile.d/cuda.sh
RUN /bin/bash -c "source /etc/profile"
RUN rm ./install/cuda_12.2.0_535.54.03_linux.run

# 安装vscode
# 先创建一个assets文件夹，把vscode上传到oss上，再下载到这个assets文件夹里
RUN mkdir asserts

RUN wget http://auto-car-preprocess.oss-cn-zhangjiakou.aliyuncs.com/lansheng/code-server_4.23.1_amd64.deb -P ./asserts/

RUN dpkg -i ./asserts/code-server_4.23.1_amd64.deb 
# && rm asserts/code-server_4.1.0_amd64.deb

COPY ./torch-2.1.0+cu121-cp38-cp38-linux_x86_64.whl ./install/
RUN conda install -y python=3.8
RUN pip install ./install/torch-2.1.0+cu121-cp38-cp38-linux_x86_64.whl
RUN rm ./install/torch-2.1.0+cu121-cp38-cp38-linux_x86_64.whl

RUN pip install --no-deps lpips==0.1.4
RUN pip install numpy==1.24.4
RUN pip install scipy==1.10.1
RUN pip install torchvision==0.16.0
RUN pip install tqdm opencv-python multiprocess opencv-python-headless plyfile jaxtyping jaxtyping open3d einops tensorboard wandb colorama imageio ninja
 
COPY ./torch_scatter-2.1.2+pt21cu121-cp38-cp38-linux_x86_64.whl ./install/
RUN pip install ./install/torch_scatter-2.1.2+pt21cu121-cp38-cp38-linux_x86_64.whl
RUN rm ./install/torch_scatter-2.1.2+pt21cu121-cp38-cp38-linux_x86_64.whl

RUN rm /etc/apt/sources.list
COPY ./sources.list /etc/apt/
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx 
RUN apt-get install -y libglib2.0
RUN apt-get install -y git 


RUN pip install multiprocess pyrender
# 添加一个用户组admin，并添加系统用户admin，赋予编号505
RUN groupadd -r -g 505 admin && useradd --no-log-init -m -r -g 505 -u 505 admin -s /bin/bash -p admin && mkdir -p /data && chown -fR admin:admin /data && echo admin:admin | chpasswd
RUN adduser admin sudo
USER root 

RUN apt-get install -y expect
ENTRYPOINT ["tail", "-f", "/dev/null"]
