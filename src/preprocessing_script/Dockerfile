FROM python:3.6

WORKDIR /usr/src/app

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN apt install unzip
RUN unzip awscliv2.zip
RUN ./aws/install
RUN aws --no-sign-request s3 cp s3://acoustic-sandbox/labeled-data/detection/train/OrcasoundLab07052019_PodCastRound2.tar.gz ./
RUN aws --no-sign-request s3 cp s3://acoustic-sandbox/labeled-data/detection/train/podcast2.tsv ./
RUN tar -xzf OrcasoundLab07052019_PodCastRound2.tar.gz

COPY preprocess.py .
COPY selection_table.py .
COPY requirements.txt . 

RUN apt-get update && apt-get install -y \
ffmpeg libavcodec-extra
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update &&  apt-get install -y \
libsndfile1-dev
RUN pip install pysndfile

ENTRYPOINT ["python3","./preprocess.py"]




