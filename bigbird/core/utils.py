# Copyright 2020 The BigBird Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helper and utility functions."""

import re

from absl import logging
import numpy as np
import tensorflow as tf


############################### SHAPE UTILS ####################################


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if not tf.executing_eagerly() and name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  assert False, "Static shape not available for {}".format(tensor)

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if not tf.executing_eagerly() and name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, int):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.compat.v1.get_variable_scope().name
    raise ValueError(
        "For the tensor `{}` in scope `{}`, the actual rank "
        "`{}` (shape = {}) is not equal to the expected rank `{}`".format(
            name, scope_name, actual_rank, str(tensor.shape),
            str(expected_rank)))


############################### DENSE LAYERS ###################################


def create_initializer(initializer_range=0.02):
  """Creates a `truncated_normal_initializer` with the given range."""
  return tf.compat.v1.truncated_normal_initializer(stddev=initializer_range)


class Dense3dLayer(tf.compat.v1.layers.Layer):
  """A dense layer with 3D kernel."""

  def __init__(self,
               num_attention_heads,
               size_per_head,
               initializer,
               activation,
               name=None,
               head_first=False,
               use_bias=True):
    """Constructor for dense layer with 3D kernel.

    Args:
      num_attention_heads: The size of output dimension.
      size_per_head: The size per attention head.
      initializer: Kernel initializer.
      activation: Actication function.
      name: The name scope of this layer.
      head_first: Whether to output head dimension before or after sequence dim.
      use_bias: Whether the layer uses a bias vector.
    """
    super(Dense3dLayer, self).__init__(name=name)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.initializer = initializer
    self.activation = activation
    self.head_first = head_first
    self.use_bias = use_bias

    self.w = None
    self.b = None

  def operation(self, input_tensor):
    """Constructor for dense layer with 3D kernel.

    Args:
      input_tensor: float Tensor of shape [batch, seq_length, hidden_size].

    Returns:
      float logits Tensor.
    """
    last_dim = get_shape_list(input_tensor)[-1]
    if self.w is None: # input_tensor * self.w -> 계산
      self.w = tf.compat.v1.get_variable( # w variable 생성
          name="kernel", # 이름 지정
          shape=[last_dim, self.num_attention_heads * self.size_per_head], # shape 생성 [768, 786]
          initializer=self.initializer)
      self.initializer = None
      self._trainable_weights.append(self.w)
    reshape_w = tf.reshape( # w variable reshape 부분
        self.w, [last_dim, self.num_attention_heads, self.size_per_head]) # [768, 12, 64]
    if self.head_first: # decoder는 first head   설정
      ret = tf.einsum("abc,cde->adbe", input_tensor, reshape_w) # input_tensor * reshape_w 계산 -> [4, 12(attention multi-head), 2048, 64(head당 사이즈)]
    else: # head 가 3번째 rank에 있는 경우
      ret = tf.einsum("abc,cde->abde", input_tensor, reshape_w)

    if self.use_bias: # bias  도 같이 계산 input_tensor * w + b
      if self.b is None:
        self.b = tf.compat.v1.get_variable(
            name="bias",
            shape=[self.num_attention_heads * self.size_per_head],
            initializer=tf.zeros_initializer)
        self._trainable_weights.append(self.b)
      if self.head_first:
        reshape_b = tf.reshape(
            self.b, [1, self.num_attention_heads, 1, self.size_per_head])
      else:
        reshape_b = tf.reshape(
            self.b, [self.num_attention_heads, self.size_per_head])
      ret += reshape_b

    if self.activation is not None: # activation 여부
      return self.activation(ret)
    else:
      return ret


class Dense3dProjLayer(tf.compat.v1.layers.Layer):
  """A dense layer with 3D kernel for projection."""

  def __init__(self,
               num_attention_heads,
               size_per_head,
               initializer,
               activation,
               name=None,
               use_bias=True):
    """Constructor for dense layer with 3D kernel for projection.

    Args:
      num_attention_heads: The size of output dimension.
      size_per_head: The size per attention head.
      initializer: Kernel initializer.
      activation: Actication function.
      name: The name scope of this layer.
      use_bias: Whether the layer uses a bias vector.
    """
    super(Dense3dProjLayer, self).__init__(name=name)
    self.num_attention_heads = num_attention_heads
    self.size_per_head = size_per_head
    self.initializer = initializer
    self.activation = activation
    self.use_bias = use_bias

    self.w = None
    self.b = None

  def operation(self, input_tensor):
    """Constructor for dense layer with 3D kernel for projection.

    Args:
      input_tensor: float Tensor of shape [batch,from_seq_length,
        num_attention_heads, size_per_head].

    Returns:
      float logits Tensor.
    """
    hidden_size = self.num_attention_heads * self.size_per_head # hiden size 768
    if self.w is None: # weight 계산 부분 (768, 768)
      self.w = tf.compat.v1.get_variable(
          name="kernel",
          shape=[hidden_size, hidden_size],
          initializer=self.initializer)
      self.initializer = None
      self._trainable_weights.append(self.w)
    reshape_w = tf.reshape( # attention feature 계산을 위한 reshape (12, 64, 768)
        self.w, [self.num_attention_heads, self.size_per_head, hidden_size])
    ret = tf.einsum("BFNH,NHD->BFD", input_tensor, reshape_w) # weight 계산 -> (4, 2048, 768)

    if self.use_bias:
      if self.b is None: # bias 계산
        self.b = tf.compat.v1.get_variable(
            name="bias",
            shape=[hidden_size],
            initializer=tf.zeros_initializer)
        self._trainable_weights.append(self.b)
      ret += self.b

    if self.activation is not None: # no activation
      return self.activation(ret)
    else:
      return ret


class Dense2dLayer(tf.compat.v1.layers.Layer):
  """A dense layer with 2D kernel."""

  def __init__(self,
               output_size,
               initializer,
               activation,
               name=None,
               use_bias=True):
    """Constructor for dense layer with 2D kernel.

    Args:
      output_size: The size of output dimension.
      initializer: Kernel initializer.
      activation: Actication function.
      name: The name scope of this layer.
      use_bias: Whether the layer uses a bias vector.
    """
    super(Dense2dLayer, self).__init__(name=name)
    self.output_size = output_size
    self.initializer = initializer
    self.activation = activation
    self.use_bias = use_bias

    self.w = None
    self.b = None

  def operation(self, input_tensor):
    """Forward pass for dense layer with 2D kernel.

    Args:
      input_tensor: Float tensor with rank 3.

    Returns:
      float logits Tensor.
    """
    if self.w is None:
      last_dim = get_shape_list(input_tensor)[-1] # 마지막 dim -> 768
      self.w = tf.compat.v1.get_variable( # weight 계산 (768, 3072)
          name="kernel",
          shape=[last_dim, self.output_size],
          initializer=self.initializer)
      self.initializer = None
      self._trainable_weights.append(self.w)
    ret = tf.einsum("abc,cd->abd", input_tensor, self.w) # 결과 -> (4, 2048, 3072)

    if self.use_bias:
      if self.b is None:
        self.b = tf.compat.v1.get_variable( # bias 계산 (3072)
            name="bias",
            shape=[self.output_size],
            initializer=tf.zeros_initializer)
        self._trainable_weights.append(self.b)
      ret += self.b

    if self.activation is not None: # gelu(Gaussian Error Linear unit: Gaussian 분포를 따르는 비선형 함수) activation 사용
      return self.activation(ret)
    else:
      return ret


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tf.tanh(
      (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
  return x * cdf


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, str):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "relu":
    return tf.nn.relu
  elif act == "gelu":
    return gelu
  elif act == "tanh":
    return tf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


########################## NORM & DROPOUT LAYERS ###############################


def dropout(input_tensor, dropout_prob, training=True):
  """Perform dropout.

  Args:
    input_tensor: float Tensor.
    dropout_prob: Python float. The probability of dropping out a value (NOT of
      *keeping* a dimension as in `tf.nn.dropout`).
    training: Boolean indicating whether the call is training or inference.

  Returns:
    A version of `input_tensor` with dropout applied.
  """
  if not training or dropout_prob is None or dropout_prob == 0.0:
    return input_tensor

  output = tf.nn.dropout(input_tensor, rate=dropout_prob)
  return output


class NormLayer(tf.compat.v1.layers.Layer):
  """Replacement for contrib_layers.layer_norm."""

  def __init__(self, name="LayerNorm"):
    super(NormLayer, self).__init__(name=name)
    self.beta = None
    self.gamma = None

  def operation(self, input_tensor):
    inputs = tf.convert_to_tensor(input_tensor) # input tensor를 tensor로 변환
    inputs_shape = get_shape_list(inputs) # shape list화
    inputs_rank = len(inputs_shape) # rank 계산 : input_rank
    dtype = inputs.dtype.base_dtype # 입력된 encoder데이터의 기본 데이터 타입 추출 (float32)
    norm_axis = inputs_rank - 1 # input rank는 3이므로 list는 0으로 시작하니깐 inputs_rank에 -1 함
    params_shape = [inputs_shape[norm_axis]] # input에 마지막 shape [768]

    # Allocate parameters for the beta and gamma of the normalization.
    if self.beta is None: # nomalize를 위한 beta 계
      self.beta = tf.compat.v1.get_variable(
          "beta",
          shape=params_shape,
          dtype=dtype,
          initializer=tf.zeros_initializer(),
          trainable=True)
      self._trainable_weights.append(self.beta)
    if self.gamma is None: # nomalize를 위한 gamma계산
      self.gamma = tf.compat.v1.get_variable(
          "gamma",
          shape=params_shape,
          dtype=dtype,
          initializer=tf.ones_initializer(),
          trainable=True)
      self._trainable_weights.append(self.gamma)
    # Compute norm along last axis
    mean, variance = tf.nn.moments(inputs, [norm_axis], keepdims=True) # inputs에 마지막 axis 2 대한 평균가 분산을 구하는 부분
    # Compute layer normalization using the batch_normalization function.
    # Note that epsilon must be increased for float16 due to the limited
    # representable range.
    variance_epsilon = 1e-12 if dtype != tf.float16 else 1e-3 # variance의 epsilon 계

    # batch nomalization 계산 수식
    # nomalized_output = gamma * ( (input_x - m) / (variance + variance_epsilon) ) + beta
    outputs = tf.nn.batch_normalization(
        inputs,
        mean,
        variance,
        offset=self.beta,
        scale=self.gamma,
        variance_epsilon=variance_epsilon) # variance가 0에 가까워서 나눈 값이 0인것 을 방지하기 위해서 작은  epsilon사용
    outputs.set_shape(inputs_shape)
    return outputs


############################# EMBEDDING LAYER ##################################


class EmbeddingLayer(tf.compat.v1.layers.Layer):
  """An embedding layer."""

  def __init__(self,
               vocab_size,
               emb_dim,
               initializer,
               scale_emb=False,
               use_token_type=False,
               num_token_types=16,
               use_position_embeddings=True,
               max_position_embeddings=4096,
               dropout_prob=0.0,
               name="embeddings"):
    super(EmbeddingLayer, self).__init__(name=name)
    self.vocab_size = vocab_size
    self.emb_dim = emb_dim
    self.scale_emb = scale_emb
    self.num_token_types = num_token_types
    self.max_position_embeddings = max_position_embeddings
    self.dropout_prob = dropout_prob

    with tf.compat.v1.variable_scope(name):
      self.word_embeddings = tf.compat.v1.get_variable( # word_embeddings 기본 [vocab_size, emb_dim] = [50358, 768]
          "word_embeddings", [vocab_size, emb_dim],
          dtype=tf.float32, initializer=initializer)
      self._trainable_weights.append(self.word_embeddings) # 학슥될 tensor들을 저장

      if use_token_type: # token의 타입을 사용하는지 기본 2개의 token cls, sep 를 사용하므로 토큰 타입에 대한 embedding실
        self.token_type_table = tf.compat.v1.get_variable(
            "token_type_embeddings", [num_token_types, emb_dim], # [2, 768]
            dtype=tf.float32, initializer=initializer)
        self._trainable_weights.append(self.token_type_table)
      else:
        self.token_type_table = None

      if use_position_embeddings: #position embedding 실행 여부(최대 Position embedding은 4096)
        self.position_embeddings = tf.compat.v1.get_variable(
            "position_embeddings", [max_position_embeddings, emb_dim], # [4096, 768]
            dtype=tf.float32, initializer=initializer)
        self._trainable_weights.append(self.position_embeddings)
      else:
        self.position_embeddings = None

  def operation(self,
           input_ids,
           seq_length,
           start_pos=0,
           token_type_ids=None,
           training=None):
    if input_ids is None:
      return None

    # subtoken embedding
    output = tf.nn.embedding_lookup(params=self.word_embeddings, ids=input_ids) # input에 대한 input embedding 반환 계산 shape(4, 2048, 768)

    if self.scale_emb: # embedding값을 scaling할 것인지
      output = output * self.emb_dim ** 0.5

    if self.token_type_table is not None: # token type embedding실행
      # This vocab will be small so we always do one-hot here, since it is
      # always faster for a small vocabulary.(!! token type 이 2개이므로 vocab가 작다, 따라서 embedding look up을 하지않고, 바로 tensordot을 이용해서 계산
      one_hot_ids = tf.one_hot(token_type_ids, depth=self.num_token_types)
      token_type_embeddings = tf.tensordot(
          one_hot_ids, self.token_type_table, 1) # (4, 2048, 768) tensor dot빠름(첫번째 one_host_ids 의 마치막에 1axis와 self.token_type_table에 첫번째 1 axis 계
      output += token_type_embeddings # input empedding을 더함

    if self.position_embeddings is not None:
      # assert_op = tf.compat.v1.assert_less_equal(
      #     start_pos + seq_length, self.max_position_embeddings)
      # with tf.control_dependencies([assert_op]):
      # So `position_embeddings` is effectively an embedding table for
      # position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(self.position_embeddings, [start_pos, 0], # position_embedding에서 position에 필요한 embedding만큼 잘라온다.
                                     [seq_length, self.emb_dim]) # size가 [2048, 768]이므로 2048, 768개만 position_embedding에서 가져온다
      expanded_position_embedding = tf.expand_dims(position_embeddings, axis=0) # axis 0에 대해서 matrix embedding 추가 shape를 확장한 -> 기존 token, type embedding에 더하기 위해서
      output += expanded_position_embedding # position embedding을 더함( batch에 모든 output_embedding 변수에 똑같은 Position embedding을 더해서 상대 위치적 계산을함)

    if training and self.dropout_prob > 0:
      output = tf.nn.dropout(output, self.dropout_prob)
    return output

  def linear(self, x):
    """Computes logits by running x through a linear layer.

    Args:
      x: A float32 tensor with shape [..., hidden_size]
    Returns:
      float32 tensor with shape [..., vocab_size].
    """
    with tf.compat.v1.name_scope("presoftmax_linear"):
      logits = tf.tensordot(x, self.word_embeddings, [[-1], [1]])
    return logits


########################## TPU/CHECKPOINT UTILS ################################


def get_estimator(config, model_fn, keep_checkpoint_max=10):
  """Create TPUEstimator object for given config and model_fn."""
  tpu_cluster_resolver = None # TPU를 사용할 것인지를 확인
  if config["use_tpu"] and config["tpu_name"]:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        config["tpu_name"],
        zone=config["tpu_zone"],
        project=config["gcp_project"])

  # Batch size book-keeping
  # Estimators handle batch sizes differently among GPUs and TPUs
  # GPU: Estimator needs per core batch size
  # TPU: Estimator needs total batch size, i.e. num_cores * per core batch size
  config_train_batch_size = config["train_batch_size"]     # For estimator
  config_eval_batch_size = config["eval_batch_size"]       # For estimator
  effective_train_batch_size = config["train_batch_size"]  # For human
  effective_eval_batch_size = config["eval_batch_size"]    # For human
  if config["use_tpu"]:  # TPU를 사용할 경우
    sliced_eval_mode = tf.compat.v1.estimator.tpu.InputPipelineConfig.SLICED
    distribute_strategy = None
    config_train_batch_size *= config["num_tpu_cores"]
    config_eval_batch_size *= config["num_tpu_cores"]
    effective_train_batch_size = config_train_batch_size
    effective_eval_batch_size = config_eval_batch_size
  else:
    sliced_eval_mode = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V1# (evaluation mode)TPU가 아니므로 입력 Pipline을 host당 하나로 정한다(PER_HOST_V1 = 1)
    distribute_strategy = tf.distribute.MirroredStrategy(devices=None) # 하나의 머신에서 여러 복제품 Device를 동기화 트레이닝 하기 위함, Multi GPU 환경을 위해서, CPU환경은 하나
    effective_train_batch_size *= distribute_strategy.num_replicas_in_sync # 멀티 GPU가 없고, CPU하나라면 num_replicas_in_sync는 1임
    # effective_eval_batch_size *= distribute_strategy.num_replicas_in_sync

  is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2 # V2= 3, per_host가 3
  run_config = tf.compat.v1.estimator.tpu.RunConfig( # TPU 실행을 support 할 config
      cluster=tpu_cluster_resolver, # TPU사용을 위한 Resolver 설정, TPU가 없으면개 None type
      master=config["master"], # training에 사용할 Master의 Address(String 값) None type
      model_dir=config["output_dir"], # model의 output dir
      save_checkpoints_steps=config["save_checkpoints_steps"], # 설정값이 100이라면 매번 100 Step마다 check point 생성 -> 200설정
      keep_checkpoint_max=keep_checkpoint_max, # 최대 max point 설정(만약 2라면 최대 2개까지 모델 check point가 저장된다.)
      train_distribute=distribute_strategy, # multi process 설정
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig( #TPU관련 설정
          tpu_job_name=config["tpu_job_name"], # TPU Job의 이름 설
          iterations_per_loop=config["iterations_per_loop"], # 하나의 세션의 Global step 이 설정된 값 만큼 증가하여 Loop가 실행, 보통 save check point step과 같음 -> 1000 값
          num_shards=config["num_tpu_cores"], # TPU core만큼 시스템의 복제품을 만듬 -> 복제품 코어 * 복제품이 병렬 학습환경에서 총 Core수 -> 8로 설정
          per_host_input_for_training=is_per_host, # 학습 단계에서 호스트 설정 PER_HOST_V2 -> 3
          eval_training_input_configuration=sliced_eval_mode)) #평가 단계에서 호스트 설정 PER_HOST_V1 -> 2

  if config["init_checkpoint"]: # Checkpoint가 설정된 경우( 이미 Trainig 된 모델이 있다면)
    ckpt_var_list = tf.compat.v1.train.list_variables(config["init_checkpoint"])
    ckpt_var_list = {
        name: shape for name, shape in ckpt_var_list
        if not re.findall("(Adam|Adafactor|global_step)", name)
    }
    vars_to_warm_start = "({})".format("|".join(ckpt_var_list.keys()))
    warm_start_settings = tf.estimator.WarmStartSettings(
        ckpt_to_initialize_from=config["init_checkpoint"],
        vars_to_warm_start=vars_to_warm_start)
  else:
    ckpt_var_list = {}
    warm_start_settings = None # warm start setting으로 학습 할 것인지
  config["ckpt_var_list"] = ckpt_var_list # check point가 있는 경우

  # If no TPU, this will fall back to normal Estimator on CPU or GPU.
  estimator = tf.compat.v1.estimator.tpu.TPUEstimator( #TPU를 지원하는 Support Estimator를 생성
      use_tpu=config["use_tpu"], # tpu 사용하지 않음
      model_fn=model_fn, # 모델을 설정
      config=run_config, # 학습할 run config 설정
      train_batch_size=config_train_batch_size,
      eval_batch_size=config_eval_batch_size,
      warm_start_from=warm_start_settings)

  # assign batch sizes
  estimator.train_batch_size = effective_train_batch_size
  estimator.eval_batch_size = effective_eval_batch_size

  return estimator

class LogVariable():
    def __init__(self, variables, ckpt_var_list):
        """Log trainable variables."""
        logging.info("**** Trainable Variables ****")
        model_var_list = {var.name: var.get_shape().as_list() for var in variables}
        num_params = sum(np.prod(shape) for shape in model_var_list.values())
        length = max(len(name) for name in model_var_list) + 2
        line = "{{:<{}}}{{:<13}}{{}}".format(length)

        logging.info("The model has {} trainable variables "
                     "({:,} parameters):\n".format(len(model_var_list), num_params))
        logging.info(line.format("Name", "Initialized", "Shape"))
        logging.info(line.format("----", "-----------", "-----"))

        ckpt_var_list = ckpt_var_list.copy()
        for name, shape in model_var_list.items():
            name = name.split(":")[0]
            if name in ckpt_var_list:
                warm_started = "from ckpt"
                del ckpt_var_list[name]
            else:
                warm_started = "random"
            logging.info(line.format(name, warm_started, shape))

        if ckpt_var_list:
            logging.warning(
                "The warm start checkpoint contained %d variables that were not used "
                "for the model:\n", len(ckpt_var_list))
            for name, shape in ckpt_var_list.items():
                logging.warning(line.format(name, "not used", shape))


def add_scalars_to_summary(summary_dir, scalar_tensors_dict):
  """Creates a host_call function that writes summaries on TPU."""

  #  All tensors outfed from TPU should preserve batch size dimension.
  scalar_tensors_dict = {
      k: tf.reshape(v, [1]) for k, v in scalar_tensors_dict.items()
  }

  def host_call_fn(**kwargs):
    writer = tf.summary.create_file_writer(summary_dir, max_queue=1000)
    always_record = tf.summary.record_if(True)
    with writer.as_default(), always_record:
      for name, scalar in kwargs.items():
        tf.summary.scalar(name, tf.reduce_mean(scalar),
                          tf.compat.v1.train.get_or_create_global_step())
      return tf.compat.v1.summary.all_v2_summary_ops()

  return host_call_fn, scalar_tensors_dict


########################## DEFAULT CONFIG UTILS ################################


def get_default_config():
  """Default values for BigBird."""

  default_config = {
      # transformer basic configs
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.1,
      "hidden_size": 768,
      "initializer_range": 0.02,
      "intermediate_size": 3072,
      "max_position_embeddings": 4096,
      "num_attention_heads": 12,
      "num_hidden_layers": 12,
      "type_vocab_size": 2,
      "use_bias": True,
      "rescale_embedding": False,
      "scope": "bert",
      # sparse mask configs
      "attention_type": "block_sparse",
      "norm_type": "postnorm",
      "block_size": 16,
      "num_rand_blocks": 3,
      # common bert configs
      "max_encoder_length": 1024,
      "max_decoder_length": 64,
      "couple_encoder_decoder": False,
      "beam_size": 5,
      "alpha": 0.7,
      "label_smoothing": 0.1,
      "weight_decay_rate": 0.01,
      "optimizer_beta1": 0.9,
      "optimizer_beta2": 0.999,
      "optimizer_epsilon": 1e-6,
      # TPU settings
      "use_tpu": True,
      "tpu_name": None,
      "tpu_zone": None,
      "tpu_job_name": None,
      "gcp_project": None,
      "master": None,
      "num_tpu_cores": 8,
      "iterations_per_loop": "1000",
  }

  return default_config
