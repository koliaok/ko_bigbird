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

"""BigBird Encoder Layers."""

from bigbird.core import attention
from bigbird.core import utils
import tensorflow as tf


class PrenormEncoderLayer(tf.compat.v1.layers.Layer):
  """Encoder layer of a transformer in Pegasus style.

  The layer_norm is taken before self-attention.
  """

  def __init__(self,
               attention_type,
               hidden_size=768,
               intermediate_size=3072,
               intermediate_act_fn=utils.gelu,
               attention_probs_dropout_prob=0.0,
               hidden_dropout_prob=0.1,
               initializer_range=0.02,
               num_attention_heads=12,
               num_rand_blocks=3,
               block_size=64,
               use_bias=True,
               seed=None,
               name=None):
    """Constructor of an encoder layer of a transformer in Pegasus style.

    Args:
      attention_type: Type of attention, needs to be one of ['original_full',
        'simulated_sparse', 'block_sparse'].
      hidden_size: (optional) int. Size of hidden dimension.
      intermediate_size: (optional) int. Size of intermediate dimension.
      intermediate_act_fn: optional) Activation function for intermediate layer.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      hidden_dropout_prob: (optional) float. Dropout probability of the
        attention.
      initializer_range: (optional) float. Range of the weight initializer.
      num_attention_heads: (optional) int. Number of attention heads.
      num_rand_blocks: (optional) int. Number of random chunks per row.
      block_size: (optional) int. size of block in sequence.
      use_bias: (optional) bool. Whether key/query/value uses a bias vector.
      seed: (Optional) int. Reandom seed for generating random mask.
      name: The name scope of this layer.
    """
    super(PrenormEncoderLayer, self).__init__(name=name)
    self.hidden_dropout_prob = hidden_dropout_prob

    # Attention layer
    attention_head_size = hidden_size // num_attention_heads
    self.attn_layer = attention.MultiHeadedAttentionLayer(
        attention_type, num_attention_heads, num_rand_blocks,
        attention_head_size, initializer_range, block_size, block_size,
        attention_probs_dropout_prob, use_bias, seed, name="self")

    # Dense layers
    self.projection_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size,
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    self.expand_layer = utils.Dense2dLayer(
        intermediate_size, utils.create_initializer(initializer_range),
        intermediate_act_fn, "dense")
    self.contract_layer = utils.Dense2dLayer(
        hidden_size, utils.create_initializer(initializer_range),
        None, "dense")

    # Normalization layer
    self.first_layer_norm = utils.NormLayer()
    self.second_layer_norm = utils.NormLayer()

  @property
  def trainable_weights(self):
    tvar_list = (self.attn_layer.trainable_weights +
                 self.projection_layer.trainable_weights +
                 self.expand_layer.trainable_weights +
                 self.contract_layer.trainable_weights +
                 self.first_layer_norm.trainable_weights +
                 self.second_layer_norm.trainable_weights)
    self._trainable_weights = list({v.name: v for v in tvar_list}.values())
    return self._trainable_weights

  def operation(self,
           layer_input,
           attention_mask=None,
           band_mask=None,
           from_mask=None,
           to_mask=None,
           input_blocked_mask=None,
           training=None):
    """Implements a encoder layer of a transformer in Pegasus style.

    Args:
      layer_input: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        seq_length, seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      band_mask: (optional) int32 Tensor of shape [batch_size, 1,
        seq_length//block_size-4, block_size, 3*block_size].
        The values should be 1 or 0. The attention scores will effectively be
        set to -infinity for any positions in the mask that are 0, and will be
        unchanged for positions that are 1.
      from_mask: (optional) int32 Tensor of shape [batch_size, 1,
        seq_length, 1]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      to_mask: (optional) int32 Tensor of shape [batch_size, 1, 1,
        seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      input_blocked_mask: (optional) int32 Tensor of shape [batch_size,
        seq_length//block_size, block_size]. Same as from/to_mask, just
        reshaped.
      training: Boolean indicating whether the call is training or inference.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size].

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
      NotImplementedError: For unknown attention type.
    """

    with tf.compat.v1.variable_scope("attention"):
      with tf.compat.v1.variable_scope("self") as sc:
        normalized_layer_input = self.first_layer_norm(layer_input)
        attention_output = self.attn_layer.operation(
            normalized_layer_input, normalized_layer_input,
            attention_mask, band_mask, from_mask, to_mask,
            input_blocked_mask, input_blocked_mask, training, scope=sc)

      # Run a linear projection of `hidden_size` then add a residual
      # with `layer_input`.
      with tf.compat.v1.variable_scope("output"):
        attention_output = self.projection_layer.operation(attention_output)
        attention_output = utils.dropout(attention_output,
                                         self.hidden_dropout_prob,
                                         training)
        attention_output = attention_output + layer_input

    # The activation is only applied to the "intermediate" hidden layer.
    with tf.compat.v1.variable_scope("intermediate"):
      normalized_attention_output = self.second_layer_norm.operation(attention_output)
      intermediate_output = self.expand_layer.operation(normalized_attention_output)

    # Down-project back to `hidden_size` then add the residual.
    with tf.compat.v1.variable_scope("output"):
      layer_output = self.contract_layer.operation(intermediate_output)
      layer_output = utils.dropout(layer_output,
                                   self.hidden_dropout_prob,
                                   training)
      layer_output = layer_output + attention_output
    return layer_output


class PostnormEncoderLayer(tf.compat.v1.layers.Layer):
  """Encoder layer of a transformer in BERT style.

  The layer_norm is taken after self-attention.
  """

  def __init__(self,
               attention_type,
               hidden_size=768,
               intermediate_size=3072,
               intermediate_act_fn=utils.gelu,
               attention_probs_dropout_prob=0.0,
               hidden_dropout_prob=0.1,
               initializer_range=0.02,
               num_attention_heads=12,
               num_rand_blocks=3,
               block_size=64,
               use_bias=True,
               seed=None,
               name=None):
    """Constructor of an encoder layer of a transformer in BERT style.

    Args:
      attention_type: Type of attention, needs to be one of ['original_full',
        'simulated_sparse', 'block_sparse'].
      hidden_size: (optional) int. Size of hidden dimension.
      intermediate_size: (optional) int. Size of intermediate dimension.
      intermediate_act_fn: optional) Activation function for intermediate layer.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      hidden_dropout_prob: (optional) float. Dropout probability of the
        attention.
      initializer_range: (optional) float. Range of the weight initializer.
      num_attention_heads: (optional) int. Number of attention heads.
      num_rand_blocks: (optional) int. Number of random chunks per row.
      block_size: (optional) int. size of block in sequence.
      use_bias: (optional) bool. Whether key/query/value uses a bias vector.
      seed: (Optional) int. Reandom seed for generating random mask.
      name: The name scope of this layer.
    """
    super(PostnormEncoderLayer, self).__init__(name=name)
    self.hidden_dropout_prob = hidden_dropout_prob

    # Attention layer의 정
    attention_head_size = hidden_size // num_attention_heads # 12 multi-head attention 을 위해서 head size를 정의
    self.attn_layer = attention.MultiHeadedAttentionLayer(
        attention_type, num_attention_heads, num_rand_blocks, # block_sparse, 12, 3
        attention_head_size, initializer_range, block_size, block_size, # 64, 0.01, 16, 16
        attention_probs_dropout_prob, use_bias, seed, name="self") # 0.01, true, (0~11 seed encoder layer에 만큼 커짐)

    # Dense layers: attention 결과를 1)추출 -> 2)확장 -> 3)축소 하는 방식으로 Feature를 더 정교하게 뽑아내는 과정
    # 1) 어텐션을 projection 하는 레이어
    self.projection_layer = utils.Dense3dProjLayer(
        num_attention_heads, attention_head_size, # 12, 64
        utils.create_initializer(initializer_range), None, "dense", use_bias)
    # 2) 확장 레이어 정의
    self.expand_layer = utils.Dense2dLayer(
        intermediate_size, utils.create_initializer(initializer_range),
        intermediate_act_fn, "dense")
    # 3) 축소 레이어 정의
    self.contract_layer = utils.Dense2dLayer( # 마지막 레이어 feature를 뽑아내는 레이어
        hidden_size, utils.create_initializer(initializer_range),
        None, "dense")

    # Normalization layer
    self.first_layer_norm = utils.NormLayer()
    self.second_layer_norm = utils.NormLayer()

  @property
  def trainable_weights(self):
    tvar_list = (self.attn_layer.trainable_weights +
                 self.projection_layer.trainable_weights +
                 self.expand_layer.trainable_weights +
                 self.contract_layer.trainable_weights +
                 self.first_layer_norm.trainable_weights +
                 self.second_layer_norm.trainable_weights)
    self._trainable_weights = list({v.name: v for v in tvar_list}.values())
    return self._trainable_weights

  def operation(self,
           layer_input,
           attention_mask=None,
           band_mask=None,
           from_mask=None,
           to_mask=None,
           input_blocked_mask=None,
           training=None):
    """Implements a encoder layer of a transformer in BERT style.

    Args:
      layer_input: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        seq_length, seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      band_mask: (optional) int32 Tensor of shape [batch_size, 1,
        seq_length//block_size-4, block_size, 3*block_size].
        The values should be 1 or 0. The attention scores will effectively be
        set to -infinity for any positions in the mask that are 0, and will be
        unchanged for positions that are 1.
      from_mask: (optional) int32 Tensor of shape [batch_size, 1,
        seq_length, 1]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      to_mask: (optional) int32 Tensor of shape [batch_size, 1, 1,
        seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      input_blocked_mask: (optional) int32 Tensor of shape [batch_size,
        seq_length//block_size, block_size]. Same as from/to_mask, just
        reshaped.
      training: Boolean indicating whether the call is training or inference.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size].

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
      NotImplementedError: For unknown attention type.
    """

    with tf.compat.v1.variable_scope("attention"):
      with tf.compat.v1.variable_scope("self") as sc:
        # 1) attnetion 계산
        attention_output = self.attn_layer.operation(
            layer_input, layer_input,
            attention_mask, band_mask, from_mask, to_mask,
            input_blocked_mask, input_blocked_mask, training, scope=sc)

      # Run a linear projection of `hidden_size` then add a residual
      # with `layer_input`.
      # 2) linear projection 계산
      with tf.compat.v1.variable_scope("output"):
        attention_output = self.projection_layer.operation(attention_output) # linear projection 실행
        attention_output = utils.dropout(attention_output, # drop out 계산
                                         self.hidden_dropout_prob,
                                         training)
        attention_output = self.first_layer_norm.operation(attention_output + layer_input) # residual 계산하고, nomalize

    # The activation is only applied to the "intermediate" hidden layer. network 확장
    with tf.compat.v1.variable_scope("intermediate"): # 여기서 activation Gelu 사
      intermediate_output = self.expand_layer.operation(attention_output)

    # Down-project back to `hidden_size` then add the residual.
    with tf.compat.v1.variable_scope("output"): # 확장된 네트워크 다시 축소
      layer_output = self.contract_layer.operation(intermediate_output)
      layer_output = utils.dropout(layer_output, # drop out 0.1
                                   self.hidden_dropout_prob,
                                   training)
      layer_output = self.second_layer_norm.operation(layer_output + attention_output)# 다시 residual 및 normalize
    return layer_output # 결과 (4, 2048, 786)


class EncoderStack(tf.compat.v1.layers.Layer):
  """Transformer encoder stack."""

  def __init__(self, params):
    name = "encoder"
    super(EncoderStack, self).__init__(name=name)
    self.params = params

    if params["norm_type"] == "prenorm": # layer norm type을 설졍
      encoder_class = PrenormEncoderLayer
    elif params["norm_type"] == "postnorm": # 기본 postnorm encoder 사
      encoder_class = PostnormEncoderLayer
    else:
      raise NotImplementedError(
          "Norm type {} is not implemented".format(params["norm_type"]))

    # Encoder layers
    self.encoder_layers = [
        encoder_class(  # pylint: disable=g-complex-comprehension
            self.params["attention_type"], # block_sparse attention type 설정
            self.params["hidden_size"], # 768
            self.params["intermediate_size"], # intermediate_size
            utils.get_activation(self.params["hidden_act"]), # gelu activation function
            self.params["attention_probs_dropout_prob"], # 0.1
            self.params["hidden_dropout_prob"], # 0.1
            self.params["initializer_range"], # 0.02
            self.params["num_attention_heads"], # num_attention_heads
            self.params["num_rand_blocks"], # rand block : 3
            self.params["block_size"], # 16
            self.params["use_bias"], # True
            seed=layer_idx,
            name="layer_%d" % layer_idx)
        for layer_idx in range(self.params["num_hidden_layers"]) # 개 encoder 12개를 list에 담음
    ]

    # Normalization layer
    self.layer_norm = utils.NormLayer()

  @property
  def trainable_weights(self):
    tvar_list = sum(
        [layer.trainable_weights for layer in self.encoder_layers],
        []) +  self.layer_norm.trainable_weights
    self._trainable_weights = list({v.name: v for v in tvar_list}.values())
    return self._trainable_weights

  def operation(self,
           encoder_inputs,
           encoder_inputs_mask,
           training=None):
    """Return the output of the decoder layer stacks.

    Args:
      encoder_inputs: tensor with shape
        [batch_size, input_length, hidden_size]
      encoder_inputs_mask: Mask for enccoder input. [batch_size, input_length]
      training: Boolean indicating whether the call is training or inference.

    Returns:
      Finaly layer encoder output. float tensor with shape
        [batch_size, input_length, hidden_size]
    """
    encoder_shape = utils.get_shape_list(encoder_inputs, expected_rank=3) # token embedding input의 shape를 list 형태로 return
    batch_size = encoder_shape[0] # batch size :4
    encoder_length = encoder_shape[1] # encoder shape: 2048

    if self.params["attention_type"] == "block_sparse": # sparse attention 일때
      # reshape and cast for blocking
      encoder_block_size = self.params["block_size"] #encoder block 사이즈 : 16 -> fine-tuning 에서는 65
      blocked_encoder_mask = tf.reshape( # encoder mask의 블록이 16이므로 input_mask를 reshape
          encoder_inputs_mask,
          (batch_size, encoder_length//encoder_block_size, encoder_block_size))
      encoder_from_mask = tf.reshape(encoder_inputs_mask, # 마스크로 부터 encoder input 값(4, 1, 2048, 1)
                                     (batch_size, 1, encoder_length, 1))
      encoder_to_mask = tf.reshape(encoder_inputs_mask, # encoder input에서 mask input으로 값(4, 1, 1, 2048)
                                   (batch_size, 1, 1, encoder_length))

      # create band padding
      attention_mask = None # attention mask만들기
      band_mask = attention.create_band_mask_from_inputs( # attention mask를 어떻게 만드는지 확인 할 필요
          blocked_encoder_mask, blocked_encoder_mask)

    else:
      blocked_encoder_mask = None
      encoder_to_mask = None
      encoder_from_mask = None

      attention_mask = attention.create_attention_mask_from_input_mask(
          encoder_inputs_mask, encoder_inputs_mask)
      band_mask = None

    # if self.params["use_gradient_checkpointing"]:
    #   encoder_layer = recompute_gradient(encoder_layer)

    if self.params["norm_type"] == "postnorm": #현재 기본 설정 Postnorm ( input을 입력하기 전 먼저 normalizer를 수행한다)
      encoder_inputs = self.layer_norm.operation(encoder_inputs) # (4, 2048, 768)의 입력을 Nomalize한다.
    # 12 layer attention 계산
    layer_output = encoder_inputs
    for layer in self.encoder_layers:
      layer_output = layer.operation(
          layer_output, attention_mask, band_mask,
          encoder_from_mask, encoder_to_mask, blocked_encoder_mask, training)

    if self.params["norm_type"] == "prenorm": # post norm type 설정(실행되지 않음) -> encoding은 prenorm type이므로 마지막에 실행
      layer_output = self.layer_norm.operation(layer_output)

    return layer_output # 모든 attention계산 결과
