# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Create masked LM/next sentence masked_lm TF examples for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from bigbird.core import flags
from absl import app
from absl import logging

import collections
import random
import tensorflow as tf
import tokenization
import numpy as np


FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_integer("max_seq_length", 2048, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 75,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_integer("split_output_data_len", 6, "split output data len")

class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, tokens_postion_ids, segment_ids,
               masked_lm_positions, masked_lm_labels, masked_lm_weights,
               is_random_next):
    self.tokens = tokens
    self.tokens_postion_ids = tokens_postion_ids
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels
    self.masked_lm_weights = masked_lm_weights


def write_instance_to_example_files(instances, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.io.TFRecordWriter(output_file))

  writer_index = 0
  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(instance.tokens)
    features["input_mask"] = create_int_feature(instance.tokens_postion_ids)
    features["segment_ids"] = create_int_feature(instance.segment_ids)
    features["masked_lm_positions"] = create_int_feature(instance.masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(instance.masked_lm_labels)
    features["masked_lm_weights"] = create_float_feature(instance.masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([instance.is_random_next])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1


  for writer in writers:
    writer.close()

  logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


def create_training_instances(input_files, tokenizer, max_seq_length,
                              masked_lm_prob, max_predictions_per_seq, rng,
                              write_instance_to_example_files, output_files):
  """Create `TrainingInstance`s from raw text."""

  def numpy_masking(subtokens):
    """
    numpy를 이용한 token 전처리 및 MLM 작업 진행
    """

    end_pos = max_seq_length - 2 + np.random.randint(
        max(1, len(subtokens) - max_seq_length - 2))
    # Find a random span in text
    start_pos = max(0, end_pos - max_seq_length + 2)
    subtokens = subtokens[start_pos:end_pos]

    # The start might be inside a word so fix it
    # such that span always starts at a word
    word_begin_mark = tokenizer.word_start_subtoken[subtokens]# Start token이 "▁" 인 token 만 True
    word_begins_pos = np.flatnonzero(word_begin_mark).astype(np.int32)# word_begin_mark에 False된 부분을 제외한 Index만 기록
    if word_begins_pos.size == 0:
      # if no word boundary present, we do not do whole word masking
      # and we fall back to random masking.
      word_begins_pos = np.arange(len(subtokens), dtype=np.int32)# subtoken만큼 word token을 만들고 np.arrange(3, dtype=np.int32) -> (0, 1, 2)
      word_begin_mark = np.logical_not(word_begin_mark)# index가 0인 것은 true  아닌것은 false
      print(subtokens, start_pos, end_pos, word_begin_mark)
    correct_start_pos = word_begins_pos[0]# 앞에 "▁" 있는 첫번째 토큰의Index를 가져옴
    subtokens = subtokens[correct_start_pos:]# 앞에 "▁" 있는 첫번째 token부터 나머지 모든 토큰을 가져옴
    word_begin_mark = word_begin_mark[correct_start_pos:]# True, False로 구성된 word_begin_mark array 첫번째 index token부터 나머지 모든 토큰을 가져옴
    word_begins_pos = word_begins_pos - correct_start_pos# 모든 array에 첫번째 Index를 뺸다
    num_tokens = len(subtokens)#sub token의 길이

    # @e want to do whole word masking so split by word boundary
    words = np.split(np.arange(num_tokens, dtype=np.int32), word_begins_pos)[1:]#word 마스킹을 위해서 "▁"로 시작하는 token의 flag index array씩 array를 Split한
    assert len(words) == len(word_begins_pos)#나누어진 words와 word_begin_pos의

    # Decide elements to mask
    num_to_predict = min(
        max_predictions_per_seq,
        max(1, int(round(len(word_begins_pos) * masked_lm_prob)))) # 최대 예측될 masking token선택, 최소 1개
    masked_lm_positions = np.concatenate(np.random.choice(# words의 array list를 최대 masking예측 개수 만큼 random하게 선택하고, 모두 concate
        np.array([[]] + words, dtype=np.object)[1:],
        num_to_predict, replace=False), 0)

    # max_predictions_per_seq보다 maked_lm_postion의 ndarray의 length가 긴 경우 다시 max_prediction_per_seq 길이 만큼 잘라줌
    if len(masked_lm_positions) > max_predictions_per_seq:
      masked_lm_positions = masked_lm_positions[:max_predictions_per_seq + 1]
      # however last word can cross word boundaries, remove crossing words
      truncate_masking_at = np.flatnonzero(
        word_begin_mark[masked_lm_positions])[-1] #마지막 flag가 True인 index를 가져옴
      masked_lm_positions = masked_lm_positions[:truncate_masking_at]

    # sort masking positions
    masked_lm_positions = np.sort(masked_lm_positions)# masked token index 오름차순 정렬
    masked_lm_ids = subtokens[masked_lm_positions]# 정렬된 token index에 각 word의 사전 index를 추출

    # replace input token with [MASK] 80%, random 10%, or leave it as it is.
    randomness = np.random.rand(len(masked_lm_positions))# uniform distribution [0, 1]사이 값을 making index array 길이 만큼 ndarray로 뽑아
    mask_index = masked_lm_positions[randomness < 0.8]# 0.8 보다 작은 값만 masking index로 가져옴
    random_index = masked_lm_positions[randomness > 0.9]# 0.9 보다 큰 값의 index는 random하게 가져

    subtokens[mask_index] = 67  # id of masked token [MASK] = 67 subtokens에 0.8보다 작은 확률의 index를 masking token으로 변경
    random_token_value = np.random.randint(  # voca에 0~100사이의 값은 special tokens 이므로 low=101, high=vocab_size로 사용하여 random index개수 만큼 random vocab index를 넣어
        101, tokenizer.vocab_size, len(random_index), dtype=np.int32)
    subtokens[random_index] = random_token_value

    # add [CLS] (65) and [SEP] (66) tokens, masking과 random으로 구성된 subtoken 앞, 뒤에 [CLS]와 [SEP] 토큰으로 채워줌
    subtokens = np.concatenate([
        np.array([65], dtype=np.int32), subtokens,
        np.array([66], dtype=np.int32)
    ])

    # pad everything to correct shape
    pad_inp = max_seq_length - num_tokens - 2# padding token의 길이 계산
    subtokens = np.pad(subtokens, [0, pad_inp], "constant")# padding token (0 ~ 2017) 만큼 <pad> token으로 처리하기
    subtokens_mask_ids = np.where(subtokens > 0, 1, 0)# 전체 토큰중에 어떤게 전처리 되었는지 마스킹
    pad_out = max_predictions_per_seq - len(masked_lm_positions)#pad가 되지 않은 sequence length를 계
    masked_lm_weights = np.pad(
        np.ones_like(masked_lm_positions, dtype=np.float32),# masked_lm_postion index를 모두 1로 변경하고 나머지를 0으로 padding한다. 여기서 type이 float이란점 기억
        [0, pad_out], "constant")
    masked_lm_positions = np.pad(
        masked_lm_positions + 1, [0, pad_out], "constant")# [CLS]토큰이 추가 되었으므로 masking position에 1을 더하고, 나머지는 padding 처리, +1은 position 한칸씩 밀어준경우
    masked_lm_ids = np.pad(masked_lm_ids, [0, pad_out], "constant")# 실제 masking 된 index값에 padding처
    segment_ids = np.ones_like(subtokens)# subtokens의 segment_ids를 계산
    next_sentence_labels = 0# next sentence prediction을 사용하지 않으므로 0

    # subtokens = masking 된 subtoken에 padding (size: max_seq_length, type: int),
    # subtokens_mask_ids = masking 된 subtoken에 (1, 0)으로 구 (size: max_seq_length, type: int),
    # masked_lm_possitions = subtokens에 masking 된 곳의 index (size: max_predictions_per_seq, type: int)
    # masked_lm_ids = masked_lm_positions에 해당되는 vocab사전 index, 학습에서는 label이 됨 (size: max_predictions분_per_seq, type: int)
    # masked_lm_weights = masked_lm_postions에 마스킹 weigths (size: max_predictions_per_seq, type: float)
    # segment_ids = subtokens의 segment_ids를 계산 (size: max_seq_length, type: int)
    # next_sentence_labels = next sentence prediction을 사용하지 않으므로 [0] 생성 (size : 1, type: int)
    return subtokens, subtokens_mask_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights, segment_ids, next_sentence_labels

  output_file = output_files[0]

  output_file = output_file.split('/')

  first_output_path = '/'.join(output_file[:-1]) + '/'

  another_output_path =output_file[-1].split('.')

  first_output_text = first_output_path + another_output_path[0]

  second_output_type = '.' + another_output_path[1]

  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.

  for input_file in input_files:
    with tf.io.gfile.GFile(input_file, "r") as reader:

      total_data = reader.readlines()

      split_data_len = len(total_data) // FLAGS.split_output_data_len

      last_data_len = len(total_data) % FLAGS.split_output_data_len

      cnt = 0

      text_cnt = 0

      for data in total_data:

        line = tokenization.convert_to_unicode(data)

        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        else:
          all_documents[-1].append(line)

        cnt += 1

        if cnt == split_data_len:
          print('number of data', cnt)

          cnt = 0
          text_cnt += 1

          # Remove empty documents
          all_documents = [x for x in all_documents if x]
          rng.shuffle(all_documents)

          instances = []

          for documents in all_documents:
            subtokens = tokenizer.tf_tokenizer.tokenize(documents)

            #tokenize된 객체 들구오기
            insert_tokens = subtokens.numpy()[0]

            # token masking처리
            (subtokens, subtokens_postion_ids, masked_lm_positions, masked_lm_ids,
             masked_lm_weights, segment_ids, next_sentence_labels) = numpy_masking(insert_tokens)

            instance = TrainingInstance(
              tokens=subtokens,
              tokens_postion_ids=subtokens_postion_ids,
              segment_ids=segment_ids,
              is_random_next=next_sentence_labels,
              masked_lm_positions=masked_lm_positions,
              masked_lm_labels=masked_lm_ids,
              masked_lm_weights=masked_lm_weights)

            instances.extend([instance])

          rng.shuffle(instances)

          output_data_name = first_output_text + '_' + str(text_cnt) + second_output_type

          write_instance_to_example_files(instances, [output_data_name])

          del all_documents

          all_documents = [[]]

          if text_cnt == (FLAGS.split_output_data_len):
            split_data_len += last_data_len
            print(split_data_len)




def main(_):

  # 1. tokenizer 생성
  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file)

  # 2. 입력 데이터 처리
  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))
  #2.1 input data 전처리전 데이터
  logging.info("*** Reading from input files ***")
  for input_file in input_files:
    logging.info("  %s", input_file)

  # 3. output_file 처리
  output_files = FLAGS.output_file.split(",")
  logging.info("*** Writing to output files ***")
  for output_file in output_files:
    logging.info("  %s", output_file)

  # 4. 학습 데이터 전처리 및 마스킹
  rng = random.Random(FLAGS.random_seed)
  create_training_instances(
    input_files, tokenizer, FLAGS.max_seq_length,
    FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
    rng, write_instance_to_example_files, output_files)


if __name__ == "__main__":
  app.run(main)
