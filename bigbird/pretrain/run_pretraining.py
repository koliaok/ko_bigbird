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

"""Run masked LM/next sentence pre-training for BigBird."""

import os
import time

from absl import app
from absl import logging
from bigbird.core import flags
from bigbird.core import modeling
from bigbird.core import optimization
from bigbird.core import utils
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft
import sentencepiece as spm
import cProfile
import collections
import re
#from tensorflow.python import debug as tf_debug

#pydevd.settrace()
FLAGS = flags.FLAGS

## Required parameters

flags.DEFINE_string(
    "data_dir", "../dataset",
    "The input data dir. Should contain the TFRecord files. "
    "Can be TF Dataset with prefix tfds://")

flags.DEFINE_string(
    "output_dir", "./tmp/bigb",
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BigBird model).")

flags.DEFINE_integer(
    "max_encoder_length", 1024,
    "The maximum total input sequence length after SentencePiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 75,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_float(
    "masked_lm_prob", 0.15,
    "Masked LM probability.")

flags.DEFINE_string(
    "substitute_newline", " ",
    "Replace newline charachter from text with supplied string.")

flags.DEFINE_bool(
    "do_train", True,
    "Whether to run training.")

flags.DEFINE_bool(
    "do_eval", False,
    "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_export", False,
    "Whether to export the model as TF SavedModel.")

flags.DEFINE_integer(
    "train_batch_size", 4,
    "Local batch size for training. "
    "Total batch size will be multiplied by number gpu/tpu cores available.")

flags.DEFINE_integer(
    "eval_batch_size", 4,
    "Local batch size for eval. "
    "Total batch size will be multiplied by number gpu/tpu cores available.")

flags.DEFINE_string(
    "optimizer", "AdamWeightDecay",
    "Optimizer to use. Can be Adafactor, Adam, and AdamWeightDecay.")

flags.DEFINE_float(
    "learning_rate", 1e-4,
    "The initial learning rate for Adam.")

flags.DEFINE_integer(
    "num_train_steps", 100000,
    "Total number of training steps to perform.")

flags.DEFINE_integer(
    "num_warmup_steps", 10000,
    "Number of steps to perform linear warmup.")

flags.DEFINE_integer(
    "save_checkpoints_steps", 1000,
    "How often to save the model checkpoint.")

flags.DEFINE_integer(
    "max_eval_steps", 100,
    "Maximum number of eval steps.")

flags.DEFINE_bool(
    "preprocessed_data", False,
    "Whether TFRecord data is already tokenized and masked.")

flags.DEFINE_bool(
    "use_nsp", False,
    "Whether to use next sentence prediction loss.")

flags.DEFINE_integer(
    "batch_size", 1,
    "batch_size"
    "batch_size")

def input_fn_builder(data_dir, vocab_model_file, masked_lm_prob,
                     max_encoder_length, max_predictions_per_seq,
                     preprocessed_data, substitute_newline, is_training,
                     tmp_dir=None):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  sp_model = spm.SentencePieceProcessor()
  sp_proto = tf.io.gfile.GFile(vocab_model_file, "rb").read()
  sp_model.LoadFromSerializedProto(sp_proto)
  vocab_size = sp_model.GetPieceSize()
  word_start_subtoken = np.array(
      [sp_model.IdToPiece(i)[0] == "▁" for i in range(vocab_size)])
  word_to_token = np.array( #사전 가저 오는 부분
      [sp_model.IdToPiece(i) for i in range(vocab_size)])

  feature_shapes = {
      "input_ids": [max_encoder_length],
      "segment_ids": [max_encoder_length],
      "masked_lm_positions": [max_predictions_per_seq],
      "masked_lm_ids": [max_predictions_per_seq],
      "masked_lm_weights": [max_predictions_per_seq],
      "next_sentence_labels": [1]
  }

  def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    name_to_features = {
        "input_ids":
            tf.io.FixedLenFeature([max_encoder_length], tf.int64),
        "segment_ids":
            tf.io.FixedLenFeature([max_encoder_length], tf.int64),
        "masked_lm_positions":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.io.FixedLenFeature([1], tf.int64),
    }
    example = tf.io.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"] # 학습 batch사이즈 설정

    # Load dataset and handle tfds separately
    split = "train" if is_training else "test" #학습인지 테스트 인지 파일 선택을 위해

    input_files = tf.io.gfile.glob(os.path.join(data_dir, "*{}.tfrecord*".format(split)))

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files)) # 전처리된 입력 데이터를 tensor dataset으로 만든다
        d = d.repeat() # repeat에 들어간 인자만큼 데이터를 넣
        d = d.shuffle(buffer_size=len(input_files)) # 입력된 파일개수만큼 버퍼를 랜덤하게 shuffle해준

        # Non deterministic mode means that the interleaving is not exact.
        # This adds even more randomness to the training pipeline.
        d = d.interleave(tf.data.TFRecordDataset,# 하나 이상의 데이터가 있을 수 있으므로 이 데이터를 상호 배피한다.
                         deterministic=False, #비결정적으로 데이터를 생성할지 여부(false : 결정적으로 생성)
                         num_parallel_calls=tf.data.experimental.AUTOTUNE) # 비동기적으로 input file을 병렬 처리하기 위한 thread pool 개수 선택/tf.data.experimental.AUTOTUNE(입력 파이프라인 설정을 위한 API, AUTOTUNE이면 자동 -1 값을 가짐)
    else:
        d = tf.data.TFRecordDataset(input_files)

    if preprocessed_data: # 이미 전처리된 데이터라면 바로 decode_record하
      d = d.map(_decode_record,
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
      d = d.shuffle(buffer_size=10000, reshuffle_each_iteration=True) # reshuffle_each_iteration: 데이터를 썩을때 buffer 사이즈 한칸씩 넘어가는데 순서를 다르게 함
      d = d.repeat()

    d = d.padded_batch(batch_size, feature_shapes,
                       drop_remainder=True)  # For static shape drop_remainder = True 마지막 배치사이즈가 배치 사이즈보다 작으면 그 데이터 셋을 drop시켜버
    return d

  return input_fn


def serving_input_fn_builder(batch_size, max_encoder_length,
                             vocab_model_file, substitute_newline):
  """Creates an `input_fn` closure for exported SavedModel."""
  def dynamic_padding(inp, min_size):
    pad_size = tf.maximum(min_size - tf.shape(inp)[1], 0)
    paddings = [[0, 0], [0, pad_size]]
    return tf.pad(inp, paddings)

  def input_fn():
    # text input
    text = tf.compat.v1.placeholder(tf.string, [batch_size], name="input_text")

    # text tokenize
    tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(vocab_model_file, "rb").read())
    if substitute_newline:
      text = tf.strings.regex_replace(text, "\n", substitute_newline)
    ids = tokenizer.tokenize(text)
    if isinstance(ids, tf.RaggedTensor):
      ids = ids.to_tensor(0)

    # text padding: Pad only if necessary and reshape properly
    padded_ids = dynamic_padding(ids, max_encoder_length)
    ids = tf.slice(padded_ids, [0, 0], [batch_size, max_encoder_length])

    receiver_tensors = {"input": text}
    features = {"input_ids": tf.cast(ids, tf.int32, name="input_ids")}

    return tf.estimator.export.ServingInputReceiver(
        features=features, receiver_tensors=receiver_tensors)

  return input_fn


def model_fn_builder(bert_config):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # BigBird model  정의
    model = modeling.BertModel(bert_config,
                               features["input_ids"],
                               training=is_training,
                               token_type_ids=features.get("segment_ids")
                               )
    # attention feature와 cls token에 대한 pooling feature를 가져옴
    sequence_output, pooled_output = model.get_output_feature()

    masked_lm = MaskedLMLayer( # masked language output 계산 모델 정의
        bert_config["hidden_size"], bert_config["vocab_size"], model.embeder,
        input_tensor=sequence_output, label_ids=features.get("masked_lm_ids"),
        label_weights=features.get("masked_lm_weights"),
        masked_lm_positions=features.get("masked_lm_positions"),
        initializer=utils.create_initializer(bert_config["initializer_range"]),
        activation_fn=utils.get_activation(bert_config["hidden_act"]))
    masked_lm_loss, masked_lm_log_probs = masked_lm.get_mlm_loss()

    """
    next_sentence = NSPLayer( # next sentence output 계산 모델 정의
        bert_config["hidden_size"],
        input_tensor=pooled_output,
        next_sentence_labels=features.get("next_sentence_labels"),
        initializer=utils.create_initializer(bert_config["initializer_range"]))
    next_sentence_loss, next_sentence_log_probs = next_sentence.get_next_sentence_loss()
    """
    total_loss = masked_lm_loss

    """
    if bert_config["use_nsp"]:
      total_loss += next_sentence_loss
    """

    tvars = tf.compat.v1.trainable_variables()
    utils.LogVariable(tvars, bert_config["ckpt_var_list"])

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      # optimize 계산
      opt_model = optimization.LinearWarmupLinearDecay( # optimize model 불러옴
          init_lr=bert_config["learning_rate"],
          num_train_steps=bert_config["num_train_steps"],
          num_warmup_steps=bert_config["num_warmup_steps"])
      learning_rate = opt_model.get_learning_rate() # laernin rate 가져옴

      optimizer = optimization.Optimizer(bert_config, learning_rate)
      optimizer = optimizer.get_optimizer()

      global_step = tf.compat.v1.train.get_global_step()

      gradients = optimizer.compute_gradients(total_loss, tvars)
      train_op = optimizer.apply_gradients(gradients, global_step=global_step)
      logging_hook = [tf.compat.v1.train.LoggingTensorHook({"loss is -> ": total_loss}, every_n_iter=32)]

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks=logging_hook,
          host_call=utils.add_scalars_to_summary(
              bert_config["output_dir"], {"learning_rate": learning_rate}))

    elif mode == tf.estimator.ModeKeys.EVAL:
      """
            def metric_fn(masked_lm_loss_value, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_loss_value,
                    next_sentence_log_probs, next_sentence_labels):
        
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.compat.v1.metrics.mean(
            values=masked_lm_loss_value)

        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_accuracy = tf.compat.v1.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.compat.v1.metrics.mean(
            values=next_sentence_loss_value)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_loss, masked_lm_log_probs, features["masked_lm_ids"],
          features["masked_lm_weights"], next_sentence_loss,
          next_sentence_log_probs, features["next_sentence_labels"]
      ])
      """
      def metric_fn(masked_lm_loss_value, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights):
        
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        masked_lm_accuracy = tf.compat.v1.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.compat.v1.metrics.mean(
            values=masked_lm_loss_value)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_loss, masked_lm_log_probs, features["masked_lm_ids"],
          features["masked_lm_weights"]
      ])

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics)
    else:
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
              "log-probabilities": masked_lm_log_probs,
              "seq-embeddings": sequence_output
          })

    return output_spec

  return model_fn


class MaskedLMLayer(tf.compat.v1.layers.Layer):
  """Get loss and log probs for the masked LM."""

  def __init__(self,
               hidden_size,
               vocab_size,
               embeder,
               input_tensor,
               initializer=None,
               activation_fn=None,
               name="cls/predictions",
               label_ids=None,
               label_weights=None,
               masked_lm_positions=None
               ):
    super(MaskedLMLayer, self).__init__(name=name)
    self.hidden_size = hidden_size # 768 사이즈 정의
    self.vocab_size = vocab_size # 50358 사전 사이즈 정
    self.embeder = embeder # embedding layer 정의

    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    self.extra_layer = utils.Dense2dLayer( # gelu activation function 사용하여 non-linear 사용
        hidden_size, initializer,
        activation_fn, "transform")
    self.norm_layer = utils.NormLayer("transform") # nomalize layer정의

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    # biase 변수 생성
    self.output_bias = tf.compat.v1.get_variable(
        name+"/output_bias",
        shape=[vocab_size], # vocab size 50358
        initializer=tf.zeros_initializer())

    if masked_lm_positions is not None: # mask lm position의 input tensor 만큼 gather하
        input_tensor = tf.gather(input_tensor, masked_lm_positions, batch_dims=1)

    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf.compat.v1.variable_scope("transform") as sc:
        input_tensor = self.extra_layer(input_tensor, scope=sc) # linear transform하고 gelu activation func 사용 (4, 75, 768)
        input_tensor = self.norm_layer(input_tensor, scope=sc) # normalize 실행

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    logits = self.embeder.linear(input_tensor) # output은 embedding weight vocab이므로 결과 -> (4, 75, 768) -> (4, 75, 50358)
    logits = tf.nn.bias_add(logits, self.output_bias) # bias 더하
    self.log_probs = tf.nn.log_softmax(logits, axis=-1) # log softmax 실행

    if label_ids is not None:
      one_hot_labels = tf.one_hot(
          label_ids, depth=self.vocab_size, dtype=tf.float32) # one-hot label 을 만든다 vocab size만

      # The `positions` tensor might be zero-padded (if the sequence is too
      # short to have the maximum number of predictions). The `label_weights`
      # tensor has a value of 1.0 for every real prediction and 0.0 for the
      # padding predictions.
      per_example_loss = -tf.reduce_sum(self.log_probs * one_hot_labels, axis=-1) # loss 구함
      numerator = tf.reduce_sum(label_weights * per_example_loss) # label weight만 계산하기 위해서
      denominator = tf.reduce_sum(label_weights) + 1e-5 # weight 합을 구함
      self.loss = numerator / denominator # 평균 구하기
    else:
      self.loss = tf.constant(0.0)


  @property
  def trainable_weights(self):
    self._trainable_weights = (self.extra_layer.trainable_weights +
                               self.norm_layer.trainable_weights +
                               [self.output_bias])
    return self._trainable_weights

  def get_mlm_loss(self):

    return self.loss, self.log_probs


class NSPLayer(tf.compat.v1.layers.Layer):
  """Get loss and log probs for the next sentence prediction."""

  def __init__(self,
               hidden_size,
               input_tensor,
               next_sentence_labels=None,
               initializer=None,
               name="cls/seq_relationship"):
    super(NSPLayer, self).__init__(name=name)
    self.hidden_size = hidden_size

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.compat.v1.variable_scope(name): #  weight 변수 선언
      self.output_weights = tf.compat.v1.get_variable(
          "output_weights",
          shape=[2, hidden_size],
          initializer=initializer)
      self._trainable_weights.append(self.output_weights)
      self.output_bias = tf.compat.v1.get_variable( # bias 변수 선언
          "output_bias", shape=[2], initializer=tf.zeros_initializer())
      self._trainable_weights.append(self.output_bias)

      logits = tf.matmul(input_tensor, self.output_weights, transpose_b=True)
      logits = tf.nn.bias_add(logits, self.output_bias)
      self.log_probs = tf.nn.log_softmax(logits, axis=-1)

      if next_sentence_labels is not None:
          labels = tf.reshape(next_sentence_labels, [-1])
          one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
          per_example_loss = -tf.reduce_sum(one_hot_labels * self.log_probs, axis=-1)
          self.loss = tf.reduce_mean(per_example_loss)
      else:
          self.loss = tf.constant(0.0)

  def get_next_sentence_loss(self):
    return self.loss, self.log_probs


def main(_):

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_export:
    raise ValueError(
        "At least one of `do_train`, `do_eval` must be True.")

  bert_config = flags.as_dictionary()

  if FLAGS.max_encoder_length > bert_config["max_position_embeddings"]:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_encoder_length, bert_config["max_position_embeddings"]))

  tf.io.gfile.makedirs(FLAGS.output_dir)
  if FLAGS.do_train:
    flags.save(os.path.join(FLAGS.output_dir, "pretrain.config"))

  model_fn = model_fn_builder(bert_config)# 학습 모델을 설정

  estimator = utils.get_estimator(bert_config, model_fn)# 모델를 학습하고, 예측할 estimator 생성
  if FLAGS.do_train:
    logging.info("***** Running training *****")
    logging.info("  Batch size = %d", estimator.train_batch_size)
    logging.info("  Num steps = %d", FLAGS.num_train_steps)

    train_input_fn = input_fn_builder( #학습에 사용될 Input data 설정
        data_dir=FLAGS.data_dir,
        vocab_model_file=FLAGS.vocab_model_file,
        masked_lm_prob=FLAGS.masked_lm_prob,
        max_encoder_length=FLAGS.max_encoder_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        preprocessed_data=FLAGS.preprocessed_data,
        substitute_newline=FLAGS.substitute_newline,
        tmp_dir=os.path.join(FLAGS.output_dir, "tfds"),
        is_training=True)
    #hooks = [tf_debug.LocalCLIDebugHook(ui_type="readline")]

    estimator.train(input_fn=train_input_fn,
                    max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval: #학습된 모델 평가
    logging.info("***** Running evaluation *****")
    logging.info("  Batch size = %d", estimator.eval_batch_size)

    eval_input_fn = input_fn_builder(
        data_dir=FLAGS.data_dir,
        vocab_model_file=FLAGS.vocab_model_file,
        masked_lm_prob=FLAGS.masked_lm_prob,
        max_encoder_length=FLAGS.max_encoder_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        preprocessed_data=FLAGS.preprocessed_data,
        substitute_newline=FLAGS.substitute_newline,
        tmp_dir=os.path.join(FLAGS.output_dir, "tfds"),
        is_training=False)

    # Run continuous evaluation for latest checkpoint as training progresses.
    last_evaluated = None
    while True:
      latest = tf.train.latest_checkpoint(FLAGS.output_dir)
      if latest == last_evaluated:
        break
        if not latest:
          logging.info("No checkpoints found yet.")
        else:
          logging.info("Latest checkpoint %s already evaluated.", latest)
        time.sleep(300)
        #continue
      else:
        logging.info("Evaluating check point %s", latest)
        last_evaluated = latest

        current_step = int(os.path.basename(latest).split("-")[1])
        output_eval_file = os.path.join(
            FLAGS.output_dir, "eval_results_{}.txt".format(current_step))
        result = estimator.evaluate(input_fn=eval_input_fn,
                                    steps=FLAGS.max_eval_steps,
                                    checkpoint_path=latest)

        with tf.io.gfile.GFile(output_eval_file, "w") as writer:
          logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_export:
    logging.info("***** Running export *****")

    serving_input_fn = serving_input_fn_builder(
        batch_size=FLAGS.eval_batch_size,
        vocab_model_file=FLAGS.vocab_model_file,
        max_encoder_length=FLAGS.max_encoder_length,
        substitute_newline=FLAGS.substitute_newline)

    estimator.export_saved_model(
        os.path.join(FLAGS.output_dir, "export"), serving_input_fn)


if __name__ == "__main__":
  tf.compat.v1.disable_v2_behavior()
  tf.compat.v1.enable_resource_variables()
  app.run(main)
