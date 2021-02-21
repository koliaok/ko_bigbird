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

"""BigBird Attention Layers."""

from absl import logging
from bigbird.core import utils
import numpy as np
import tensorflow as tf


MAX_SEQ_LEN = 4096


def get_single_block_row_attention(block_id,
                                   to_start_block_id,
                                   to_end_block_id,
                                   num_rand_blocks,
                                   window_block_left=1,
                                   window_block_right=1,
                                   global_block_left=1,
                                   global_block_right=1):
  """For a single row block get random row attention.

  Args:
    block_id: int. block id of row.
    to_start_block_id: int. random attention coloum start id.
    to_end_block_id: int. random attention coloum end id.
    num_rand_blocks: int. number of random blocks to be selected.
    window_block_left: int. number of blocks of window to left of a block.
    window_block_right: int. number of blocks of window to right of a block.
    global_block_left: int. Number of blocks globally used to the left.
    global_block_right: int. Number of blocks globally used to the right.

  Returns:
    row containing the random attention vector of size num_rand_blocks.
  """

  # list of to_blocks from which to choose random attention
  to_block_list = np.arange(to_start_block_id, to_end_block_id,
                            dtype=np.int32)
  # permute the blocks
  perm_block = np.random.permutation(to_block_list) # random하게 block의 숫자를 바꾼다.
  # print(perm_block)

  # illegal blocks for the current block id, using window
  illegal_blocks = list(
      range(block_id - window_block_left, block_id + window_block_right + 1)) # window block 생성

  # Add blocks at the start and at the end
  illegal_blocks.extend(list(range(global_block_left))) # window block에 처음과 끝 index를 추가한다
  illegal_blocks.extend(
      list(range(to_end_block_id - global_block_right, to_end_block_id)))

  # The second from_block cannot choose random attention on second last to_block
  if block_id == 1:
    illegal_blocks.append(to_end_block_id-2) # block_id 가 1부터 시작이므로 뒤에서 2번째 Index값이 9를 넣는

  # The second last from_block cannot choose random attention on second to_block
  if block_id == to_end_block_id - 2:
    illegal_blocks.append(1)

  selected_random_blokcs = []

  # perm_block에서 illegal_block에 없는 index값일 경우 random index로 선택
  for i in range(to_end_block_id - to_start_block_id):
    if perm_block[i] not in illegal_blocks:
      selected_random_blokcs.append(perm_block[i])
    if len(selected_random_blokcs) == num_rand_blocks:
      break
  return np.array(selected_random_blokcs, dtype=np.int32)


def bigbird_block_rand_mask_with_head(from_seq_length,
                                      to_seq_length,
                                      from_block_size,
                                      to_block_size,
                                      num_heads,
                                      plan_from_length,
                                      plan_num_rand_blocks,
                                      window_block_left=1,
                                      window_block_right=1,
                                      global_block_top=1,
                                      global_block_bottom=1,
                                      global_block_left=1,
                                      global_block_right=1):
  """Create adjacency list of random attention.

  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_heads: int. total number of heads.
    plan_from_length: list. plan from lenght where num_rand are choosen from.
    plan_num_rand_blocks: list. number of rand blocks within the plan.
    window_block_left: int. number of blocks of window to left of a block.
    window_block_right: int. number of blocks of window to right of a block.
    global_block_top: int. number of blocks at the top.
    global_block_bottom: int. number of blocks at the bottom.
    global_block_left: int. Number of blocks globally used to the left.
    global_block_right: int. Number of blocks globally used to the right.

  Returns:
    adjacency list of size num_head where each element is of size
    from_seq_length//from_block_size-2 by num_rand_blocks
  """
  assert from_seq_length//from_block_size == to_seq_length//to_block_size, \
      "Error the number of blocks needs to be same!"

  assert from_seq_length in plan_from_length, \
      "Error from sequence length not in plan!"

  # Total number of blocks in the mmask
  num_blocks = from_seq_length//from_block_size # block 개수가 몇인지: 128 -> 2048 // 16
  # Number of blocks per plan
  plan_block_length = np.array(plan_from_length) // from_block_size # random block의 ending position에 block_size만큼 나눔 : [11, 128] -> [176, 2048]//16
  # till when to follow plan
  max_plan_idx = plan_from_length.index(from_seq_length) # 최대 plan에 인덱스 : 1
  # Random Attention adjajency list
  rand_attn = [np.zeros((num_blocks,
                         np.sum(plan_num_rand_blocks[:max_plan_idx+1])),
                        dtype=np.int32) for i in range(num_heads)] # rand_attn이 저장될 배열생성 shape: [12, 128, 3] -> 12 (multi-head  attention 12,  2048에 각 block_size 16 이므로 128, random_block은 3)

  # We will go iteratively over the plan blocks and pick random number of
  # Attention blocks from the legally allowed blocks
  for plan_idx in range(max_plan_idx+1):
    rnd_r_cnt = 0
    if plan_idx > 0: # 처음 rand attention 인가?
      # set the row for all from_blocks starting from 0 to
      # plan_block_length[plan_idx-1]
      # column indx start fromm plan_block_length[plan_idx-1] and ends at
      # plan_block_length[plan_idx]
      if plan_num_rand_blocks[plan_idx] > 0:
        rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
        curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx+1]))
        for blk_rw_idx in range(global_block_top,
                                plan_block_length[plan_idx-1]):
          for h in range(num_heads):
            # print("head", h, "blk_rw_idx", blk_rw_idx)
            rand_attn[h][blk_rw_idx,
                         rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                             block_id=blk_rw_idx,
                             to_start_block_id=plan_block_length[plan_idx - 1],
                             to_end_block_id=plan_block_length[plan_idx],
                             num_rand_blocks=plan_num_rand_blocks[plan_idx],
                             window_block_left=window_block_left,
                             window_block_right=window_block_right,
                             global_block_left=global_block_left,
                             global_block_right=global_block_right)

      for pl_id in range(plan_idx):
        if plan_num_rand_blocks[pl_id] == 0:
          continue
        for blk_rw_idx in range(plan_block_length[plan_idx-1],
                                plan_block_length[plan_idx]):
          rnd_r_cnt = 0
          to_start_block_id = 0
          if pl_id > 0:
            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
            to_start_block_id = plan_block_length[pl_id-1]
          curr_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id+1]))
          for h in range(num_heads):
            # print("head", h, "blk_rw_idx", blk_rw_idx)
            rand_attn[h][blk_rw_idx,
                         rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                             block_id=blk_rw_idx,
                             to_start_block_id=to_start_block_id,
                             to_end_block_id=plan_block_length[pl_id],
                             num_rand_blocks=plan_num_rand_blocks[pl_id],
                             window_block_left=window_block_left,
                             window_block_right=window_block_right,
                             global_block_left=global_block_left,
                             global_block_right=global_block_right)

    if plan_num_rand_blocks[plan_idx] == 0: # 처음 rand attention하는 부분 plan index 0
      continue
    # print("Start from here")
    curr_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx+1])) # random block sum -> 3
    from_start_block_id = global_block_top # global block top 값을 block id가 시작하는 변수에 넣음
    to_start_block_id = 0
    if plan_idx > 0:
      rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
      from_start_block_id = plan_block_length[plan_idx-1]
      to_start_block_id = plan_block_length[plan_idx-1]

    for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
      for h in range(num_heads): # head 12,
        # print("head", h, "blk_rw_idx", blk_rw_idx)
        rand_attn[h][blk_rw_idx,
                     rnd_r_cnt:curr_r_cnt] = get_single_block_row_attention(
                         block_id=blk_rw_idx,
                         to_start_block_id=to_start_block_id,
                         to_end_block_id=plan_block_length[plan_idx],
                         num_rand_blocks=plan_num_rand_blocks[plan_idx],
                         window_block_left=window_block_left,
                         window_block_right=window_block_right,
                         global_block_left=global_block_left,
                         global_block_right=global_block_right)

  for nh in range(num_heads): # 가장 앞과 뒤에 random block을 제외한 나머지 블락을 복사해서 사용 128-2 = 126
    rand_attn[nh] = rand_attn[nh][global_block_top:num_blocks -
                                  global_block_bottom, :] # global token index에 포함되는 0번째 block을 제외한 모든 random 블락을 복
  return rand_attn


def get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
  """Gives the plan of where to put random attention.

  Args:
    from_seq_length: int. length of from sequence.
    from_block_size: int. size of block in from sequence.
    num_rand_blocks: int. Number of random chunks per row.

  Returns:
    plan_from_length: ending location of from block
    plan_num_rand_blocks: number of random ending location for each block
  """
  # general plan
  plan_from_length = []
  plan_num_rand_blocks = []
  if (2*num_rand_blocks + 5) < (from_seq_length // from_block_size):
    plan_from_length.append(int((2*num_rand_blocks + 5)*from_block_size))# random block 이 끝나는 Position 넘버(176)
    plan_num_rand_blocks.append(num_rand_blocks) # 각 block에 random이 끝나는 위치의 숫자 저장(3)
    plan_from_length.append(from_seq_length) # (176, 2048)
    plan_num_rand_blocks.append(0) #(3,0)
  elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
    plan_from_length.append(int((num_rand_blocks + 5)*from_block_size))
    plan_num_rand_blocks.append(num_rand_blocks//2)
    plan_from_length.append(from_seq_length)
    plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks//2))
  else:
    plan_from_length.append(from_seq_length)
    plan_num_rand_blocks.append(num_rand_blocks)

  return plan_from_length, plan_num_rand_blocks


def bigbird_block_rand_mask(from_seq_length,
                            to_seq_length,
                            from_block_size,
                            to_block_size,
                            num_rand_blocks,
                            last_idx=-1):
  """Create adjacency list of random attention.

  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_rand_blocks: int. Number of random chunks per row.
    last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
      if positive then num_rand_blocks blocks choosen only upto last_idx.

  Returns:
    adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
  """
  assert from_seq_length//from_block_size == to_seq_length//to_block_size, \
      "Error the number of blocks needs to be same!"

  rand_attn = np.zeros(
      (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32) # 기본 0을 base로 하는 block attention 실행 -> 결과 (62/3)
  middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)# middle 값 (1~62) -> 결과 (62)
  last = to_seq_length // to_block_size - 1 # 4096/64 -> 15
  if last_idx > (2 * to_block_size):
    last = (last_idx // to_block_size) - 1

  r = num_rand_blocks  # shorthand
  for i in range(1, from_seq_length // from_block_size-1): # 1~65
    start = i-2
    end = i
    if i == 1:
      rand_attn[i-1, :] = np.random.permutation(middle_seq[2:last])[:r] # 2~15사이의 랜덤 값을 제사용없이 random하게 뽑고, rand attn의 index = 0에 3개 값을 넣음
    elif i == 2:
      rand_attn[i-1, :] = np.random.permutation(middle_seq[3:last])[:r]  # 3~15사이의 랜덤 값을 제사용없이 random하게 뽑고, rand attn의 index = 1에 3개 값을 넣음
    elif i == from_seq_length // from_block_size - 3: # i가 61일 때
      rand_attn[i-1, :] = np.random.permutation(middle_seq[:last])[:r] # 1~15사이의 값을 랜덤을 선택하여 3개 값을 뽑음      # Missing -3: should have been sliced till last-3
    elif i == from_seq_length // from_block_size - 2: # i가 62일 때
      rand_attn[i-1, :] = np.random.permutation(middle_seq[:last])[:r]
      # Missing -4: should have been sliced till last-4
    else:
      if start > last: # last 15 보다 start가 클
        start = last
        rand_attn[i-1, :] = np.random.permutation(middle_seq[:start])[:r] # 3개 random선택
      elif (end+1) == last: # end+1과 last가 같을 때
        rand_attn[i-1, :] = np.random.permutation(middle_seq[:start])[:r] # 1~12 사이의 값 중에 3개 random선택
      else:
        rand_attn[i-1, :] = np.random.permutation(
            np.concatenate((middle_seq[:start], middle_seq[end+1:last])))[:r] # middle_seq 의 [:start] 와 [end+1:last(15)] 배열 합에서 3개만 random하게 선
  return rand_attn


def full_bigbird_mask(from_seq_length,
                      to_seq_length,
                      from_block_size,
                      to_block_size,
                      num_rand_blocks,
                      rand_attn=None,
                      focus=1024):
  """Calculate BigBird attention pattern as a full dense matrix.

  Args:
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    num_rand_blocks: int. Number of random chunks per row.
    rand_attn: adjajency matrix for random attention.
    focus: pick random mask within focus

  Returns:
    attention mask matrix of shape [from_seq_length, to_seq_length]
  """
  if rand_attn is None:
    rand_attn = bigbird_block_rand_mask(MAX_SEQ_LEN, MAX_SEQ_LEN,
                                        from_block_size, to_block_size,
                                        num_rand_blocks, focus)

  attn_mask = np.zeros((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=np.int32)
  for i in range(1, (MAX_SEQ_LEN // from_block_size) - 1):
    attn_mask[(i) * from_block_size:(i + 1) * from_block_size,
              (i - 1) * to_block_size:(i + 2) * to_block_size] = 1
    for j in rand_attn[i - 1, :]:
      attn_mask[i * from_block_size:(i + 1) * from_block_size,
                j * to_block_size:(j + 1) * to_block_size] = 1

  attn_mask[:from_block_size, :] = 1
  attn_mask[:, :to_block_size] = 1
  attn_mask[:, -to_block_size:] = 1
  attn_mask[-from_block_size:, :] = 1
  clipped_attn_mask = attn_mask[:from_seq_length, :to_seq_length]
  return np.array(clipped_attn_mask, dtype=bool)


def create_rand_mask_from_inputs(from_blocked_mask,
                                 to_blocked_mask,
                                 rand_attn,
                                 num_attention_heads,
                                 num_rand_blocks,
                                 batch_size,
                                 from_seq_length,
                                 from_block_size):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_blocked_mask: 2D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].
    rand_attn: [batch_size, num_attention_heads,
      from_seq_length//from_block_size-2, num_rand_blocks]
    num_attention_heads: int. Number of attention heads.
    num_rand_blocks: int. Number of random chunks per row.
    batch_size: int. Batch size for computation.
    from_seq_length: int. length of from sequence.
    from_block_size: int. size of block in from sequence.

  Returns:
    float Tensor of shape [batch_size, num_attention_heads,
                           from_seq_length//from_block_size-2,
                           from_block_size, num_rand_blocks*to_block_size].
  """
  num_windows = from_seq_length // from_block_size - 2 # window size 사이즈 정의 126
  random_index = tf.gather(to_blocked_mask, rand_attn, batch_dims=1)
  rand_mask = tf.reshape( # index 된 랜덤 값을 reshape
      random_index, [
          batch_size, num_attention_heads, num_windows,
          num_rand_blocks * from_block_size
      ])
  rand_mask = tf.einsum("BLQ,BHLK->BHLQK", from_blocked_mask[:, 1:-1],
                        rand_mask) # einsum 계산 결과 (4, 12, 126, 16, 48)
  return rand_mask


def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
  """Create 3D attention mask from a 2D tensor mask.

  Args:
    from_blocked_mask: 2D Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
    to_blocked_mask: int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].

  Returns:
    float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4,
                           from_block_size,  3*to_block_size].
  """

  exp_blocked_to_pad = tf.concat(
      [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2],
       to_blocked_mask[:, 3:-1]], 2)# 1 dim에 모두 4칸씩 움직여 가면 128 -> 124 block사이즈로 하여 concat, -> 결과 (4, 124, 48)

  band_mask = tf.einsum("BLQ,BLK->BLQK",
                        tf.cast(from_blocked_mask[:, 2:-2], tf.float32),
                        tf.cast(exp_blocked_to_pad, tf.float32)) # (4, 124, 16, 48) shape 계산-

  band_mask = tf.expand_dims(band_mask, 1) # (4, 1, 124, 16, 48)확장 -> band mask생성

  return band_mask


def create_attention_mask_from_input_mask(from_mask, to_mask):
  """Create attention mask from a 2D tensor mask.

  Args:
    from_mask: int32 Tensor of shape [batch_size, from_seq_length].
    to_mask: int32 Tensor of shape [batch_size, to_seq_length].

  Returns:
    int32 Tensor of shape [batch_size, 1, from_seq_length, to_seq_length].
  """
  mask = tf.einsum("BF,BT->BFT", from_mask, to_mask)

  # expand to create a slot for heads.
  mask = tf.expand_dims(mask, 1)

  return mask


def original_full_attention(query_layer,
                            key_layer,
                            value_layer,
                            attention_mask,
                            size_per_head,
                            attention_probs_dropout_prob):
  """Full quadratic attention calculation.

  Args:
    query_layer: float Tensor of shape [batch_size, num_attention_heads,
      from_seq_length, size_per_head]
    key_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    value_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    size_per_head: (optional) int. Size of each attention head.
    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].
  """

  # Directly take n^2 dot product between "query" and "key".
  attention_scores = tf.einsum("BNFH,BNTH->BNFT", query_layer, key_layer) # query, Key dot product (2, 16, 256, 64) * (2, 16, 256, 64) ->(2, 16, 256, 256)
  attention_scores = tf.multiply(attention_scores, # scaling
                                 1.0 / np.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0 # attention mask 계산

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder # 마스크 더하

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores) # softmax 계

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  attention_probs = utils.dropout(attention_probs, attention_probs_dropout_prob) # dropout 계산

  # `context_layer` = [B, F, N, H]
  context_layer = tf.einsum("BNFT,BNTH->BFNH", attention_probs, value_layer)
  return context_layer


def bigbird_simulated_attention(query_layer,
                                key_layer,
                                value_layer,
                                attention_mask,
                                num_attention_heads,
                                num_rand_blocks,
                                size_per_head,
                                from_seq_length,
                                to_seq_length,
                                from_block_size,
                                to_block_size,
                                seed=None):
  """BigBird attention calculation using masks in quadratic time.

  Args:
    query_layer: float Tensor of shape [batch_size, num_attention_heads,
      from_seq_length, size_per_head]
    key_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    value_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    attention_mask: int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    num_rand_blocks: int. Number of random chunks per row.
    size_per_head: int. Size of each attention head.
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    seed: (Optional) int. Reandom seed for generating random mask.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].
  """

  if seed:
    np.random.seed(seed)

  plan_from_length, plan_num_rand_blocks = get_rand_attn_plan(
      from_seq_length, from_block_size, num_rand_blocks)

  rand_attn = bigbird_block_rand_mask_with_head(
      from_seq_length=from_seq_length,
      to_seq_length=to_seq_length,
      from_block_size=from_block_size,
      to_block_size=to_block_size,
      num_heads=num_attention_heads,
      plan_from_length=plan_from_length,
      plan_num_rand_blocks=plan_num_rand_blocks)
  temp_mask = [
      full_bigbird_mask(  # pylint: disable=g-complex-comprehension
          from_seq_length, to_seq_length, from_block_size, to_block_size,
          num_rand_blocks, rand_attn=rand_attn[i], focus=1024)
      for i in range(num_attention_heads)
  ]
  temp_mask = np.stack(temp_mask, axis=0)
  temp_mask = np.array(temp_mask, dtype=bool)

  rand_block_mask = tf.constant(temp_mask, dtype=tf.bool)  # [N, F, T]
  rand_block_mask = tf.cast(rand_block_mask, tf.int32)
  rand_block_mask = tf.expand_dims(rand_block_mask, 0)  # [1, N, F, T]
  if attention_mask is not None:
    attention_mask = tf.minimum(attention_mask, rand_block_mask)
  else:
    attention_mask = rand_block_mask
  return original_full_attention(query_layer,
                                 key_layer,
                                 value_layer,
                                 attention_mask,
                                 size_per_head,
                                 attention_probs_dropout_prob=0.0)


def bigbird_block_sparse_attention(query_layer,
                                   key_layer,
                                   value_layer,
                                   band_mask,
                                   from_mask,
                                   to_mask,
                                   from_blocked_mask,
                                   to_blocked_mask,
                                   num_attention_heads,
                                   num_rand_blocks,
                                   size_per_head,
                                   batch_size,
                                   from_seq_length,
                                   to_seq_length,
                                   from_block_size,
                                   to_block_size,
                                   seed=None,
                                   plan_from_length=None,
                                   plan_num_rand_blocks=None):
  """BigBird attention sparse calculation using blocks in linear time.

  Assumes from_seq_length//from_block_size == to_seq_length//to_block_size.


  Args:
    query_layer: float Tensor of shape [batch_size, num_attention_heads,
      from_seq_length, size_per_head]
    key_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    value_layer: float Tensor of shape [batch_size, num_attention_heads,
      to_seq_length, size_per_head]
    band_mask: (optional) int32 Tensor of shape [batch_size, 1,
      from_seq_length//from_block_size-4, from_block_size, 3*to_block_size].
      The values should be 1 or 0. The attention scores will effectively be
      set to -infinity for any positions in the mask that are 0, and will be
      unchanged for positions that are 1.
    from_mask: (optional) int32 Tensor of shape [batch_size, 1,
      from_seq_length, 1]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    to_mask: (optional) int32 Tensor of shape [batch_size, 1, 1,
      to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    from_blocked_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length//from_block_size, from_block_size].
      Same as from_mask, just reshaped.
    to_blocked_mask: (optional) int32 Tensor of shape [batch_size,
      to_seq_length//to_block_size, to_block_size].
      Same as to_mask, just reshaped.
    num_attention_heads: int. Number of attention heads.
    num_rand_blocks: int. Number of random chunks per row.
    size_per_head: int. Size of each attention head.
    batch_size: int. Batch size for computation.
    from_seq_length: int. length of from sequence.
    to_seq_length: int. length of to sequence.
    from_block_size: int. size of block in from sequence.
    to_block_size: int. size of block in to sequence.
    seed: (Optional) int. Reandom seed for generating random mask.
    plan_from_length: (Optional) list. Plan of where to put random attn. It
      divides the block matrix into chuncks, where each chunck will have
      some randomm attn.
    plan_num_rand_blocks: (Optional) list. Number of random per block given by
      plan_from_length.

  Returns:
    float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
      size_per_head].
  """
  assert from_seq_length//from_block_size == to_seq_length//to_block_size

  # cast masks to float
  from_mask = tf.cast(from_mask, tf.float32)
  to_mask = tf.cast(to_mask, tf.float32)
  band_mask = tf.cast(band_mask, tf.float32)
  from_blocked_mask = tf.cast(from_blocked_mask, tf.float32)
  to_blocked_mask = tf.cast(to_blocked_mask, tf.float32)

  # generate random attention and corresponding masks, random attention 수
  np.random.seed(seed)
  if from_seq_length in [1024, 3072, 4096]:  # old plans used in paper #
    rand_attn = [
        bigbird_block_rand_mask(  # pylint: disable=g-complex-comprehension
            MAX_SEQ_LEN, MAX_SEQ_LEN,
            from_block_size, to_block_size, num_rand_blocks,
            last_idx=1024)[:(from_seq_length // from_block_size - 2)] # 0~(48-2)
        for _ in range(num_attention_heads)
    ]
  else:
    if plan_from_length is None:
      plan_from_length, plan_num_rand_blocks = get_rand_attn_plan( # random attention을 위한 plan값 추
          from_seq_length, from_block_size, num_rand_blocks)

    # random attention계산하는 부분 return : (12-header, (sequence_length/block_size)-2, random_number) -> (12, 126, 3)
    # -2를 하는 이유는 가장 첫번째 배열과 가장 뒤에 배열을 제거한 것이다. rank 3에 3개 배열은 random 값이 저
    rand_attn = bigbird_block_rand_mask_with_head(
        from_seq_length=from_seq_length,
        to_seq_length=to_seq_length,
        from_block_size=from_block_size,
        to_block_size=to_block_size,
        num_heads=num_attention_heads,
        plan_from_length=plan_from_length,
        plan_num_rand_blocks=plan_num_rand_blocks)

  rand_attn = np.stack(rand_attn, axis=0) #(12list , 126, 3) -> numpy(12, 126, 3)
  rand_attn = tf.constant(rand_attn, dtype=tf.int32) # numpy -> tensor 변환
  rand_attn = tf.expand_dims(rand_attn, 0) # dimenssion 추가 -> (12, 126, 3) -> (1, 12, 126, 3)
  rand_attn = tf.repeat(rand_attn, batch_size, 0) # -> (1, 12, 126, 3) -> batch사이즈 만큼 반복해서 axis=0 으로 쌓기 -> (4, 12, 126, 3)

  rand_mask = create_rand_mask_from_inputs( #random attention mask 생성
      from_blocked_mask, to_blocked_mask, rand_attn,
      num_attention_heads, num_rand_blocks,
      batch_size, from_seq_length, from_block_size,)


  # Define shorthands
  h = num_attention_heads
  r = num_rand_blocks
  d = size_per_head
  b = batch_size
  m = from_seq_length
  n = to_seq_length
  wm = from_block_size
  wn = to_block_size

  blocked_query_matrix = tf.reshape(query_layer, (b, h, m // wm, wm, -1)) # query layer (4, 12, 2048, 64) -> reshape(4, 12, 128, 16[block size], 64)
  blocked_key_matrix = tf.reshape(key_layer, (b, h, n // wn, wn, -1)) # key layer (4, 12, 2048, 64) -> reshape(4, 12, 128, 16[block size], 64)
  blocked_value_matrix = tf.reshape(value_layer, (b, h, n // wn, wn, -1)) # value layer (4, 12, 2048, 64) -> reshape(4, 12, 128, 16[block size], 64)

  # random attention index 결과인 rand_attn 값을 indices 로 사용하여 Key, Value matrix를 attention 함
  # 여기서 왜 key와 value만 계산하는가? 이런 의문점을 가지고 있어야 함 -> original query 값에서 attention 계산을 해야하니깐
  gathered_key = tf.reshape( # gether 결과 -> (4, 12, 126, 3, 16, 64) 에서 reshape 실행 -> (4, 12, 126, 48, 64)
      tf.gather(blocked_key_matrix, rand_attn, batch_dims=2, name="gather_key"),
      (b, h, m // wm - 2, r * wn, -1))  # [b, h, n//wn-2, r, wn, -1]
  gathered_value = tf.reshape(
      tf.gather(
          blocked_value_matrix, rand_attn, batch_dims=2, name="gather_value"),
      (b, h, m // wm - 2, r * wn, -1))  # [b, h, n//wn-2, r, wn, -1]

  # 첫번째 토큰에 대한 어텐션 계산 부분 original key layer 와 reshape된 query layer에 첫번째 token에 대한 값 random attention 값 계산
  # ---> first header rolling attetion
  # global token attention
  first_product = tf.einsum( # ((4, 12, 128, 16, 64) -> (4, 12, 16, 64) * (4, 12, 2048, 64)# Block size와 embedding token을 곱함
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 0],
      key_layer)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n] -> (4, 12, 16, 2048)
  first_product = tf.multiply(first_product, 1.0 / np.sqrt(d)) # sqrt -> 루트값(head size)을 나누어서 값 scaling
  first_product += (1.0 - to_mask) * -10000.0 # 마스킹 처리 (no masking 은 1, masking은 0)
  first_attn_weights = tf.nn.softmax(first_product)  # [b, h, wm, n] # attention을 위한 softmax 계산
  first_context_layer = tf.einsum( # origingal value 값과 attention weight 값 계산(전체 embedding token에 대해서 계산)
      "BHQK,BHKD->BHQD", first_attn_weights, # (4, 12, 16, 2048) * (4, 12, 2048, 64) -> (4, 12, 16, 64)
      value_layer)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
  first_context_layer = tf.expand_dims(first_context_layer, 2) # (4, 12, 16, 64) -> (4, 12, 1, 16, 64) 3번째 랭크에 dim 확장


  # 두번째 attention (-1, 0, 1, 3, random 0) 조합 계산(random 값이 적용된 Key 와 value 계산, gathered_key, gathered_value, 계산)
  # ---> left rolling attetion
  # 순서
  # 1. random계산된 값  concat index(0, 1, 2, -1, random index: 0)
  # 2. 1번 결과의 key와 기존 query 내적
  # 3. 사용될 padding tensor 만들기 (random, original mask)
  # 4. scaling
  # 5. softmax
  # 6. 5번 결과와 두번째 attention조합한 value값과 내적 및 dim expand
  second_key_mat = tf.concat([ # (4, 12, 16, 64)[0], (4, 12, 16, 64)[1], (4, 12, 16, 64)[2], (4, 12, 16, 64)[-1], (4, 12, 48, 64)[0]) concat
      blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, 1],
      blocked_key_matrix[:, :, 2], blocked_key_matrix[:, :, -1],
      gathered_key[:, :, 0]], 2)  # [b, h, (4+r)*wn, -1] -> (4, 12, 16*4+48=112, 64)
  second_value_mat = tf.concat([
      blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, 1],
      blocked_value_matrix[:, :, 2], blocked_value_matrix[:, :, -1],
      gathered_value[:, :, 0]], 2)  # [b, h, (4+r)*wn, -1]
  # query matrix 계산 (4, 12, 128, 16, 64) -> (4, 12, 16, 64) attention과 key random attention한 값에 blocked key matrix와 concat한 결과 (4, 12, 112, 64)
  second_product = tf.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, 1], second_key_mat
  )  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn] -> attention 결과 -> (4, 12, 16, 112)
  second_seq_pad = tf.concat([ # 각 sequence의 second attention padding 계
      to_mask[:, :, :, :3 * wn], to_mask[:, :, :, -wn:], # (4, 1, 1, 48)[앞 48개 token], (4, 1, 1, 16)[뒤 16개 token], (4, 1, 1, 48)(ones) 4번째 rank concat
      tf.ones([b, 1, 1, r * wn], dtype=tf.float32)], 3) #-> 결과 (4, 1, 1, 112)
  second_rand_pad = tf.concat( # second random padding 계산 random mask의 첫번째 : (4, 12, 126, 16, 48)[3번째, 0 index] , ones (4, 12, 16, 64(4*block size)) 4번째 rank concat 계산
      [tf.ones([b, h, wm, 4 * wn], dtype=tf.float32), rand_mask[:, :, 0]], 3) # 결과 -> (4, 12, 16, 112)
  second_product = tf.multiply(second_product, 1.0 / np.sqrt(d)) # query 와 key를 attention 한 부분을 scaling

  second_product += (1.0 -
                     tf.minimum(second_seq_pad, second_rand_pad)) * -10000.0 # 계산된 2개의 padding값 중에 작은 값을 선택(second_rand_pad값이 더 작음)
  second_attn_weights = tf.nn.softmax(second_product)  # [b , h, wm, (4+r)*wn]
  second_context_layer = tf.einsum( # attention한 결과를 second_value_mat와 내적  (4, 12, 16, 112) * (4, 12, 112, 64)
      "BHQK,BHKD->BHQD", second_attn_weights, second_value_mat
  )  # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1] #결과 -> (4, 12, 16, 64)
  second_context_layer = tf.expand_dims(second_context_layer, 2) # dimension 확장

  # 세번째 attention 조합 계산(random 값이 적용된 Key 와 value 계산) 중간 값 계산함 ---> middle attetion
  exp_blocked_key_matrix = tf.concat([ # original key 값 계산
      blocked_key_matrix[:, :, 1:-3], blocked_key_matrix[:, :, 2:-2], #(4, 12, 124, 16, 64), 세번째 랭크에서 크기에 4를 뺸 것을 shift 하면서 계산
      blocked_key_matrix[:, :, 3:-1]], 3)  # [b, h, m//wm-4, 3*wn, -1] # 결과값 -> (4, 12, 124, 48, 64)
  exp_blocked_value_matrix = tf.concat([ # original value 계
      blocked_value_matrix[:, :, 1:-3], blocked_value_matrix[:, :, 2:-2],
      blocked_value_matrix[:, :, 3:-1]], 3)  # [b, h, m//wm-4, 3*wn, -1]
  middle_query_matrix = blocked_query_matrix[:, :, 2:-2] #중간 query 계산 2 ~ -2 사이(from 128) 결과 값 (4, 12, 124, 16, 64)

  #내적 계산 부분
  inner_band_product = tf.einsum( # query와 중간 key의 내적 계산
      "BHLQD,BHLKD->BHLQK", middle_query_matrix, exp_blocked_key_matrix
  )  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, 3*wn, -1] -> 결과 (4, 12, 124, 48, 16) -> head 크기 만큼 attention
  rand_band_product = tf.einsum( # query 와 gathered_key(random index 에 적용된 key값) 의 내적
      "BHLQD,BHLKD->BHLQK", middle_query_matrix, gathered_key[:, :, 1:-1]
  )  # [b, h, m//wm-4, wm, -1] x [b, h, m//wm-4, r*wn, -1] 결과 ->  (4, 12, 124, 16, 48)
  first_band_product = tf.einsum( # key 의 rank 3 첫번째 값과 query값 내적
      "BHLQD,BHKD->BHLQK", middle_query_matrix, blocked_key_matrix[:, :, 0] # 첫번째 key
  )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn] -> 결과 (4, 12, 124, 16, 16)
  last_band_product = tf.einsum( # key의 rank 3 번째의 마지막 값과 qeury 값 내적
      "BHLQD,BHKD->BHLQK", middle_query_matrix, blocked_key_matrix[:, :, -1] # 마지막 key
  )  # [b, h, m//wm-4, wm, -1] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, wn]-> 결과 (4, 12, 124, 16, 16)

  # scaling 부분
  inner_band_product = tf.multiply(inner_band_product, 1.0 / np.sqrt(d)) # scaling
  rand_band_product = tf.multiply(rand_band_product, 1.0 / np.sqrt(d)) # random 값 scaling
  first_band_product = tf.multiply(first_band_product, 1.0 / np.sqrt(d)) #첫번째 key 값 scaling
  last_band_product = tf.multiply(last_band_product, 1.0 / np.sqrt(d)) # scaling

  # masking 부분
  inner_band_product += (1.0 - band_mask) * -10000.0
  first_band_product += (1.0 - tf.expand_dims(to_mask[:, :, :, :wn], 3)) * -10000.0
  last_band_product += (1.0 - tf.expand_dims(to_mask[:, :, :, -wn:], 3)) * -10000.0
  rand_band_product += (1.0 - rand_mask[:, :, 1:-1]) * -10000.0

  # softmax를 위한 concat 및 softmax 연산
  band_product = tf.concat([
      first_band_product, inner_band_product, rand_band_product,
      last_band_product], -1)  # [b, h, m//wm-4, wm, (5+r)*wn] -> 결과 (4, 12, 124, 16, 128)
  attn_weights = tf.nn.softmax(band_product)  # [b, h, m//wm-4, wm, (5+r)*wn] attention weight 계산 결과 (4, 12, 124, 16, 128)

  # 4개 value 값 attention 계산 모두 더
  context_layer = tf.einsum( # attn_weights : (4, 12, 124, 16, 128) * (4, 12, 124, 48, 64)  ->
      "BHLQK,BHLKD->BHLQD", attn_weights[:, :, :, :, wn:(4 * wn)], # -> (4, 12, 124, 16, 48)
      exp_blocked_value_matrix
  )  # [b, h, m//wm-4, wm, 3*wn] x [b, h, m//wm-4, 3*wn, -1] 결과 -> (4, 12, 124, 16, 48)
  #     ==> [b, h, m//wm-4, wm, -1]
  context_layer += tf.einsum(
      "BHLQK,BHLKD->BHLQD", attn_weights[:, :, :, :, 4 * wn:-wn], #(4, 12, 124, 16, 48(64 ~ -4)
      gathered_value[:, :, 1:-1] # (4, 12, 124, 48, 64)
  )  # [b, h, m//wm-4, wm, r*wn] x [b, h, m//wm-4, r*wn, -1] -> 결과 (4, 12, 124, 16, 64)
  #     ==> [b, h, m//wm-4, wm, -1]
  context_layer += tf.einsum(
      "BHLQK,BHKD->BHLQD", attn_weights[:, :, :, :, :wn], # (4, 12, 124, 16, 16[0~16])
      blocked_value_matrix[:, :, 0] # (4, 12, 128(0), 16, 64)
  )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1]  -> 결과 (4, 12, 124, 16, 64)
  context_layer += tf.einsum(
      "BHLQK,BHKD->BHLQD", attn_weights[:, :, :, :, -wn:], # (4, 12, 124, 16, 16[-16:0])
      blocked_value_matrix[:, :, -1]   # (4, 12, 128(-1), 16, 64)
  )  # [b, h, m//wm-4, wm, wn] x [b, h, wn, -1] ==> [b, h, m//wm-4, wm, -1] -> 결과 (4, 12, 124, 16, 64)

  # 4번째 attention : 두번째는 마지막 값 attention ( -3, -2, -1, 0, random(-1 index))  ---> right rolling attetion
  second_last_key_mat = tf.concat([ # 마지막 key값 계산
      blocked_key_matrix[:, :, 0], blocked_key_matrix[:, :, -3],
      blocked_key_matrix[:, :, -2], blocked_key_matrix[:, :, -1],
      gathered_key[:, :, -1]], 2)  # [b, h, (4+r)*wn, -1] -> key 결과 값 : (4, 12, 112, 64)
  second_last_value_mat = tf.concat([ # 마지막 value 값 계
      blocked_value_matrix[:, :, 0], blocked_value_matrix[:, :, -3],
      blocked_value_matrix[:, :, -2], blocked_value_matrix[:, :, -1],
      gathered_value[:, :, -1]], 2)  # [b, h, (4+r)*wn, -1] -> value 결과 값 : (4, 12, 112, 64)
  second_last_product = tf.einsum( # query key 내적 계산
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, -2], second_last_key_mat
  )  # [b, h, wm, -1] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, (4+r)*wn] 결과 -> (4, 12, 16, 112)
  second_last_seq_pad = tf.concat([ #마스크 계산
      to_mask[:, :, :, :wn], to_mask[:, :, :, -3 * wn:],
      tf.ones([b, 1, 1, r * wn], dtype=tf.float32)], 3) # 결과 -> (4, 1, 1, 112)
  second_last_rand_pad = tf.concat( # random  마스크 계산
      [tf.ones([b, h, wm, 4 * wn], dtype=tf.float32), rand_mask[:, :, -1]], 3)
  second_last_product = tf.multiply(second_last_product, 1.0 / np.sqrt(d)) # scaling
  second_last_product += (
      1.0 - tf.minimum(second_last_seq_pad, second_last_rand_pad)) * -10000.0 # masking작업
  second_last_attn_weights = tf.nn.softmax( # attention할 weight 계산
      second_last_product)  # [b, h, wm, (4+r)*wn]
  second_last_context_layer = tf.einsum( # value 값에 attention weight 값 게
      "BHQK,BHKD->BHQD", second_last_attn_weights, second_last_value_mat
  )  # [b, h, wm, (4+r)*wn] x [b, h, (4+r)*wn, -1] ==> [b, h, wm, -1]산 # 결과 (4, 12, 16, 64)
  second_last_context_layer = tf.expand_dims(second_last_context_layer, 2) # 결과 (4, 12, 1, 16, 64)

  # 5번째 attetnon  : index -1에 대한 attention ---> last header attetion
  last_product = tf.einsum(
      "BHQD,BHKD->BHQK", blocked_query_matrix[:, :, -1],
      key_layer)  # [b, h, wm, -1] x [b, h, n, -1] ==> [b, h, wm, n]
  last_product = tf.multiply(last_product, 1.0 / np.sqrt(d))
  last_product += (1.0 - to_mask) * -10000.0
  last_attn_weights = tf.nn.softmax(last_product)  # [b, h, wm, n]
  last_context_layer = tf.einsum(
      "BHQK,BHKD->BHQD", last_attn_weights,
      value_layer)  # [b, h, wm, n] x [b, h, n, -1] ==> [b, h, wm, -1]
  last_context_layer = tf.expand_dims(last_context_layer, 2)

  # 5개 attention (1. first header, 2. left rolling, 3. middle , 4. right rolling, 5. last head) concat
  context_layer = tf.concat([
      first_context_layer, second_context_layer, context_layer,
      second_last_context_layer, last_context_layer
  ], 2) # 결과  -> (4, 12, 128, 16, 64)
  context_layer = tf.reshape(context_layer, (b, h, m, -1)) * from_mask # 다시 마스크 적용 -> 결과 (4, 12, 2048, 64)
  context_layer = tf.transpose(context_layer, (0, 2, 1, 3)) # (4, 2048, 12, 64)
  return context_layer


class MultiHeadedAttentionLayer(tf.compat.v1.layers.Layer):
  """A multi-headed attention layer.

  It implements following types of multi-headed attention:
  - original_full attention from "Attention is all you Need".
  - simulated_sparse attention from BigBird with full quadratic implemention.
  - block_sparse attention from BigBird with memory efficient linear impl.
  """

  def __init__(self,
               attention_type,
               num_attention_heads=1,
               num_rand_blocks=3,
               size_per_head=512,
               initializer_range=0.02,
               from_block_size=64,
               to_block_size=64,
               attention_probs_dropout_prob=0.0,
               use_bias=True,
               seed=None,
               query_act=None,
               key_act=None,
               value_act=None,
               name=None,
               **kwargs):
    """Constructor for a multi-headed attention layer.

    Args:
      attention_type: Type of attention, needs to be one of ['original_full',
        'simulated_sparse', 'block_sparse'].
      num_attention_heads: (optional) int. Number of attention heads.
      num_rand_blocks: (optional) int. Number of random chunks per row.
      size_per_head: (optional) int. Size of each attention head.
      initializer_range: (optional) float. Range of the weight initializer.
      from_block_size: (optional) int. size of block in from sequence.
      to_block_size: (optional) int. size of block in to sequence.
      attention_probs_dropout_prob: (optional) float. Dropout probability of the
        attention probabilities.
      use_bias: Whether the layer uses a bias vector.
      seed: (Optional) int. Reandom seed for generating random mask.
      query_act: (optional) Activation function for the query transform.
      key_act: (optional) Activation function for the key transform.
      value_act: (optional) Activation function for the value transform.
      name: The name scope of this layer.
      **kwargs: others
    """
    super(MultiHeadedAttentionLayer, self).__init__(name=name, **kwargs)
    # query, key, value 생성
    self.query_layer = utils.Dense3dLayer( # query layer 정의
        num_attention_heads, size_per_head,
        utils.create_initializer(initializer_range), query_act,
        "query", head_first=True, use_bias=use_bias)

    self.key_layer = utils.Dense3dLayer( # key layer 정의
        num_attention_heads, size_per_head,
        utils.create_initializer(initializer_range), key_act,
        "key", head_first=True, use_bias=use_bias)

    self.value_layer = utils.Dense3dLayer( #value layer 정의
        num_attention_heads, size_per_head,
        utils.create_initializer(initializer_range), value_act,
        "value", head_first=True, use_bias=use_bias)

    def attn_impl(
        query, key, value, attention_mask,
        band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask,
        batch_size, from_seq_length, to_seq_length, training):
      if attention_type == "original_full": # decoder 사용시 256 token에 대한 summary를 하므로 original_full 사
        logging.info("**** Using original full attention ****")
        attn_fn = original_full_attention(
            query, key, value,
            attention_mask, size_per_head,
            attention_probs_dropout_prob if training else 0.0)
      elif attention_type == "simulated_sparse":
        logging.info("**** Using simulated sparse attention ****")
        attn_fn = bigbird_simulated_attention(
            query, key, value,
            attention_mask, num_attention_heads, num_rand_blocks, size_per_head,
            from_seq_length, to_seq_length, from_block_size, to_block_size,
            seed)
      elif attention_type == "block_sparse":
        logging.info("**** Using block sparse attention ****")
        attn_fn = bigbird_block_sparse_attention(
            query, key, value,
            band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask,
            num_attention_heads, num_rand_blocks, size_per_head, batch_size,
            from_seq_length, to_seq_length, from_block_size, to_block_size,
            seed)
      else:
        raise NotImplementedError(
            "Attention type {} is not implemented".format(attention_type))
      return attn_fn

    self.attn_impl = attn_impl

  @property
  def trainable_weights(self):
    tvar_list = (self.query_layer.trainable_weights +
                 self.key_layer.trainable_weights +
                 self.value_layer.trainable_weights)
    self._trainable_weights = list({v.name: v for v in tvar_list}.values())
    return self._trainable_weights

  def operation(self,
           from_tensor,
           to_tensor,
           attention_mask=None,
           band_mask=None,
           from_mask=None,
           to_mask=None,
           from_blocked_mask=None,
           to_blocked_mask=None,
           cache=None,
           decode_i=None,
           training=None,
           scope=None):
    """Implements a multi-headed attention layer from from_tensor to to_tensor.

    Args:
      from_tensor: float Tensor of shape [batch_size, from_seq_length,
        from_width]
      to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].
      attention_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length, to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      band_mask: (optional) int32 Tensor of shape [batch_size, 1,
        from_seq_length//from_block_size-4, from_block_size, 3*to_block_size].
        The values should be 1 or 0. The attention scores will effectively be
        set to -infinity for any positions in the mask that are 0, and will be
        unchanged for positions that are 1.
      from_mask: (optional) int32 Tensor of shape [batch_size, 1,
        from_seq_length, 1]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      to_mask: (optional) int32 Tensor of shape [batch_size, 1, 1,
        to_seq_length]. The values should be 1 or 0. The
        attention scores will effectively be set to -infinity for any positions
        in the mask that are 0, and will be unchanged for positions that are 1.
      from_blocked_mask: (optional) int32 Tensor of shape [batch_size,
        from_seq_length//from_block_size, from_block_size].
        Same as from_mask, just reshaped.
      to_blocked_mask: (optional) int32 Tensor of shape [batch_size,
        to_seq_length//to_block_size, to_block_size].
        Same as to_mask, just reshaped.
      cache: (Used during prediction) A dictionary with tensors containing
        results of previous attentions. The dictionary must have the items:
            {"k": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head],
             "v": tensor with shape
                  [batch_size, max_len, num_attention_heads, size_per_head]}
      decode_i: (Used during prediction) current location of decoding
      training: Boolean indicating whether the call is training or inference.

    Returns:
      float Tensor of shape [batch_size, from_seq_length, num_attention_heads,
        size_per_head].

    Raises:
      ValueError: Any of the arguments or tensor shapes are invalid.
      NotImplementedError: For unknown attention type.
    """
    self._set_scope(scope=scope) # tf.compat.v1.layers.Layer를 상속한 MultiheadAttetion 의 학습 scope 를 설
    from_shape = utils.get_shape_list(from_tensor, expected_rank=3) # 입력될 tensor shape list를 보여줌
    to_shape = utils.get_shape_list(to_tensor, expected_rank=3) # 출력될 tensor의 shape list를 보여

    if len(from_shape) != len(to_shape):# 들어오는 input값과 나가는 input값이 같은
      raise ValueError(
          "The rank of `from_tensor` must match the rank of `to_tensor`.")

    if len(from_shape) == 3:
      batch_size = from_shape[0] # batch shape 저장 : 4
      from_seq_length = from_shape[1] # sequence length 저장 : 2048 -> decoder는 256(summary)
      to_seq_length = to_shape[1] # sequence length 저장 : 2048 -> decoder는 256(summary)
    else:
      raise ValueError(
          "Need rank 3 tensors to attention_layer.")

    # Scalar dimensions referenced here:
    #   b = batch size (number of sequences)
    #   m = `from_tensor` sequence length
    #   n = `to_tensor` sequence length
    #   h = `num_attention_heads`
    #   d = `size_per_head`

    # `query` = [b, h, m, d]
    query = self.query_layer.operation(from_tensor) # query 3D layer query 계산 output tensor : (4, 12, 2048, 64), Decoder 경우 (2, 16, 3072, 64)

    # `key` = [b, h, n, d]
    key = self.key_layer.operation(to_tensor)# key 3D layer query 계산 output tensor : (4, 12, 2048, 64), Decoder 경우 (2, 16, 3072, 64)

    # `value_layer` = [b, h, n, d]
    value = self.value_layer.operation(to_tensor) # value 3D layer query 계산 output tensor : (4, 12, 2048, 64), Decoder 경우 (2, 16, 3072, 64)

    if cache is not None and decode_i is not None:
      max_len = utils.get_shape_list(cache["k"])[2] # cache 저장
      indices_select = tf.reshape(
          tf.one_hot(decode_i, max_len, dtype=to_tensor.dtype), # 각 token 에 따라서 one hot 값을 기존  cache에 입력
          [1, 1, max_len, 1])
      key = cache["k"] + key * indices_select # token에 index값을
      value = cache["v"] + value * indices_select
      cache["k"] = key
      cache["v"] = value

    # attention하는 부분
    contextual_output = self.attn_impl(
        query, key, value, attention_mask,
        band_mask, from_mask, to_mask, from_blocked_mask, to_blocked_mask,
        batch_size, from_seq_length, to_seq_length, training)
    # 입력때와 같은 shape이며, sparse attention한 결과 결과 (4, 2048, 12, 64)
    return contextual_output
