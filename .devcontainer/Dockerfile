FROM python:3.8-buster

# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo
RUN apt-get update && apt-get install -y python3-opencv

RUN echo "Install python packages"
RUN pip install numpy matplotlib seaborn opencv-python