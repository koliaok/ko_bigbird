import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft

vocab_model_file = '../bpe_model/ko_big_bird.model'
"""
dataset = tfds.load('scientific_papers/pubmed', split='test', shuffle_files=False, as_supervised=True)
# inspect at a few examples
for ex in dataset.take(3):
  print(ex)
"""

text = "삼성증권이 9일 낸 보고서 제목이다. 공매도 재개 우려가 무색하게 증시 흐름이 견조하다는 내용이다"
string_tensor = tf.constant(text, dtype=tf.string)

tokenizer = tft.SentencepieceTokenizer(
    model=tf.io.gfile.GFile(vocab_model_file, "rb").read())

ids = tokenizer.tokenize(text)
if isinstance(ids, tf.RaggedTensor):
    ids = ids.to_tensor(0)



tf.compat.v2.enable_v2_behavior()


path = '../model/serving'
imported_model = tf.saved_model.load(path, tags='serve')
summerize = imported_model.signatures['serving_default']

predicted_summary = summerize(string_tensor)

seq_embedding = predicted_summary['seq-embeddings'][0]

res = seq_embedding.numpy().tolist()
print(res)
