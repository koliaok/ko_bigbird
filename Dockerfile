FROM tensorflow/tensorflow:2.4.1-gpu

ARG TIMEZONE=Asia/Seoul
RUN ln -sf /usr/share/zoneinfo/${TIMEZONE} /etc/localtime

RUN apt-get update && apt-get upgrade -y
RUN apt-get install build-essential checkinstall -y
RUN apt-get install python3.8 -y && apt-get install vim -y

RUN python3.8 -m pip install --upgrade pip && python3.8 -m pip install poetry

WORKDIR /app/ko_bigbird

#COPY pretraining_crontab /etc/cron.d/pretraining_crontab
#RUN chmod 0644 /etc/cron.d/pretraining_crontab && crontab /etc/cron.d/pretraining_crontab
#RUN touch /var/log/pretraining_log_down.txt && touch /var/log/pretraining_log_excution.txt

COPY pyproject.toml /app/ko_bigbird/pyproject.toml
COPY poetry.lock /app/ko_bigbird/poetry.lock
RUN poetry install --no-dev

#CMD cron -f

#RUN mkdir opt
#RUN cd opt
#RUN wget https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz
#RUN tar xzf Python-3.8.1.tgz
#RUN cd Python-3.8.1
#RUN ./configure --enable-optimizations
#RUN make altinstall
