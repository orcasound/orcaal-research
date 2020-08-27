FROM python:3.8.5

WORKDIR /usr/src/app

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN apt install unzip
RUN unzip awscliv2.zip
RUN ./aws/install
RUN aws --no-sign-request s3 cp s3://acoustic-sandbox/preprocess_mag_scipy_Srkws.h5 ./
RUN aws --no-sign-request s3 cp s3://acoustic-sandbox/test_psd.zip ./
RUN unzip test_psd.zip 

COPY report.py .
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["python3","./report.py"]





