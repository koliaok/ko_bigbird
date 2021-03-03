FROM tensorflow/tensorflow:2.4.0-gpu
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN pip install poetry==1.0.5