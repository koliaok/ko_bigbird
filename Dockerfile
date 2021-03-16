FROM tensorflow/tensorflow:2.4.1-gpu

#RUN ln -sf /usr/share/zoneinfo/${SERVER_TIMEZONE} /etc/localtime
#RUN pip install poetry==1.0.5

RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install build-essential checkinstall -y
RUN apt-get install python3.8 -y
RUN apt-get install vim -y
RUN apt-get install cron -y

ENV TZ=Asia/Seoul
RUN echo $TZ > /etc/timezone && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install poetry

WORKDIR /app/ko_bigbird

COPY pyproject.toml /app/ko_bigbird/pyproject.toml
COPY poetry.lock /app/ko_bigbird/poetry.lock
RUN poetry install --no-dev

#RUN mkdir opt
#RUN cd opt
#RUN wget https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz
#RUN tar xzf Python-3.8.1.tgz
#RUN cd Python-3.8.1
#RUN ./configure --enable-optimizations
#RUN make altinstall
