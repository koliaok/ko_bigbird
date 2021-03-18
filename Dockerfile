FROM tensorflow/tensorflow:2.4.1-gpu

ARG INVESTPICK_TIMEZONE=Asia/Seoul
RUN ln -sf /usr/share/zoneinfo/${INVESTPICK_TIMEZONE} /etc/localtime


RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install build-essential checkinstall -y
RUN apt-get install python3.8 -y
RUN apt-get install vim -y
RUN apt-get install cron -y

RUN python3.8 -m pip install --upgrade pip
RUN python3.8 -m pip install poetry

WORKDIR /app/ko_bigbird

COPY pyproject.toml /app/ko_bigbird/pyproject.toml
COPY poetry.lock /app/ko_bigbird/poetry.lock
RUN poetry install --no-dev

COPY /app/ko_bigbird/bigbird/pretrain/pretraining_crontab.sh /etc/cron.d/pretraining_crontab
RUN chmod 0644 /etc/cron.d/pretraining_crontab
RUN crontab /etc/cron.d/pretraining_crontab
RUN touch /app/ko_bigbird/bigbird/pretrain/log_down.txt
RUN touch /app/ko_bigbird/bigbird/pretrain/log_excution.txt
CMD cron

#RUN mkdir opt
#RUN cd opt
#RUN wget https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz
#RUN tar xzf Python-3.8.1.tgz
#RUN cd Python-3.8.1
#RUN ./configure --enable-optimizations
#RUN make altinstall
