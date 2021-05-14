# A TensorFlow implementation of [bigbird](https://arxiv.org/abs/2007.14062) korea language version

It is korea language version bigbird training and test code repository.
I develop environment using [python poetry](https://python-poetry.org/)


## 1. data pre-processing
* data : 뉴스 기사 + 한국어 위키 데이터
* vocab type: Sentencepiece BPE
* BPE vocab size : 10000
* training data : [CLS] + document + [SEP]
* max_encoder_length: 1024


## 2. pretraining
Pre-training Model Test: loss, accuracy 


## 3. setup
* docker environment
* ubuntu 18.04
* python 3.8
* tensorflow 2.4.1 version

* need [GPU](https://blog.naver.com/wideeyed/222075635186) setting
* install docker
* install nvidia-container-toolkit
```
$ distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
$ curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
$ curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
$ sudo apt-get update
$ sudo apt-get install -y nvidia-container-toolkit
$ sudo systemctl restart docker
```
* create docker volume
```
$ docker volume create vol
$ docker volume inspect vol
```

* git clone docker volume directory
* change your volume directory in "docker-compose.yml" file
* you make tf pretrain data set and bpe model
* make BPE model
```
$ poetry install
$ poetry shell
$ cd data_preprocessing
$ python make_sentence_piece_model.py --input_data=[data directory] --output_model=[model output directory]
```

* make pretraining data and test dataset
```
$ cd bigbird/create_dataset
$ ./run_create_tf_data.sh # training data, you need to change data source directory in shell file
$ ./run_create_tf_test_data.sh # training data, you need to change data source directory in shell file
```

* change output model directory in bigbird/create_dataset/run_pretraining.sh, run_pretraining_test.sh, run_pretraining_create_serve_model.sh.sh
* you change shell file permission 
* if your completed setting, start docker compose pretrain model train, validation, serve model test
```
$ docker-compose build
$ ./docker_up.sh # start pretrain
$ ./docker_down.sh # stop docker proc
$ ./docker_up_pretraining_test.sh # pretraining model validation
$ ./docker_serve_model_test.sh # serve model attention test

```
