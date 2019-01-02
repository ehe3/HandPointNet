FROM pytorch/pytorch

WORKDIR /workspace
ADD . /workspace

RUN pip install --upgrade pip && pip install -r requirements.txt

# Download VIM
RUN apt-get update
RUN apt-get install -y vim

EXPOSE 8097