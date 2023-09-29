FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04

ENV TZ=Asia/Almaty
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    git \
    wget \
    libprotobuf-dev \
    protobuf-compiler \
    libvulkan-dev \
    vulkan-utils \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler \
    sudo \
    cmake \
    curl \
    ssh \
    rsync \
    ffmpeg \
    clang \
    ca-certificates \
    libjpeg-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    libsm6 \
    libxext6 \
    python3-dev \
    python3-pip

RUN sudo wget https://cmake.org/files/v3.25/cmake-3.25.1.tar.gz && \
    sudo tar -zxf cmake-3*.tar.gz && \
    sudo rm /cmake-3*.tar.gz && \
    cd /cmake-3* && \
    sudo ./configure --system-curl && \
    sudo make -j32&& \
    sudo make install && \
    cmake --version

RUN apt install -y \
    python3 python3-pip python3-dev python3-numpy && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade setuptools && \
    pip3 install --upgrade wheel && \
    pip3 install numpy && \
    pip3 install glob2 && \
    python3 -V

RUN wget https://github.com/opencv/opencv/archive/4.5.1.tar.gz && tar -xvf *.tar.gz && mkdir opencv-4.5.1/build \
&& cd opencv-4.5.1/build && wget https://github.com/opencv/opencv_contrib/archive/4.5.1.tar.gz && tar -xvf *.tar.gz \
&& cmake -D OPENCV_EXTRA_MODULES_PATH=opencv_contrib-4.5.1/modules .. \
&& make -j32 && make install && rm -rf /tmp/opencv-4.5.1* /tmp/*.tar.gz

RUN git clone https://github.com/Tencent/ncnn.git && cd ncnn && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DNCNN_VULKAN=OFF \
    -DNCNN_SYSTEM_GLSLANG=ON -DNCNN_BUILD_EXAMPLES=ON \
    -DNCNN_SHARED_LIB=ON .. && make -j$(nproc) && \
    make install

RUN ( \
    echo 'LogLevel DEBUG2'; \
    echo 'PermitRootLogin yes'; \
    echo 'PasswordAuthentication yes'; \
    echo 'Subsystem sftp /usr/lib/openssh/sftp-server'; \
  ) > /etc/ssh/sshd_config_test_clion \
  && mkdir /run/sshd

RUN useradd -m user \
  && yes password | passwd user && adduser user sudo
#USER user

RUN usermod -s /bin/bash user

CMD ["/usr/sbin/sshd", "-D", "-e", "-f", "/etc/ssh/sshd_config_test_clion"]