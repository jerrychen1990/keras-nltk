FROM registry-vpc.cn-hangzhou.aliyuncs.com/eigenlab/yudexcutor:tf1.12
COPY requirements.txt /opt/yud/cache/r1.txt
RUN /opt/anaconda/anaconda3/bin/pip install -r /opt/yud/cache/r1.txt
RUN mkdir pony
WORKDIR /pony/eigen-nlp-toolkit
ADD . /pony/eigen-nlp-toolkit
