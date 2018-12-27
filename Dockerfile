FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV LANG "ja_JP.UTF-8"
ENV PYTHONIOENCODING "utf-8"
ARG N_JOBS=16

RUN apt update -y && \
    apt install -y wget cmake pkg-config libprotobuf9v5 protobuf-compiler \
                   libprotobuf-dev libgoogle-perftools-dev \
                   python3 python3-dev python3-pip \
                   language-pack-ja build-essential

# install packages required in BERT
WORKDIR /tmp/library

COPY ./requirements.txt /tmp/library
RUN pip3 install -r requirements.txt

# install sentencepiece and wikiextractor
WORKDIR /tmp/library/sentencepiece/build

COPY ./library /tmp/library
RUN cmake .. && make -j ${N_JOBS} && make install
RUN ldconfig -v

WORKDIR /tmp/library/wikiextractor
RUN pip3 install -e .

WORKDIR /work
