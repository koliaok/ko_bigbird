import os
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tft
#data_dir="./tmp/bigb/tfds/wiki40b/en/1.3.0/"
data_dir = "../summarization/summary_output/tfds/scientific_papers/pubmed/1.1.1"

substitute_newline = '<n>'
vocab_model_file = '/Users/hyungrakkim/Desktop/ko_bigbird/bigbird/vocab/pegasus.model'
max_encoder_length = 3072
max_decoder_length = 256

def _tokenize_example(document, summary):
    tokenizer = tft.SentencepieceTokenizer(
        model=tf.io.gfile.GFile(vocab_model_file, "rb").read())
    if substitute_newline:
        document = tf.strings.regex_replace(document, "\n", substitute_newline)
    # Remove space before special tokens.
    document = tf.strings.regex_replace(document, r" ([<\[]\S+[>\]])", b"\\1")
    document_ids = tokenizer.tokenize(document)
    if isinstance(document_ids, tf.RaggedTensor):
        document_ids = document_ids.to_tensor(0)
    document_ids = document_ids[:max_encoder_length]

    # Remove newline optionally
    if substitute_newline:
        summary = tf.strings.regex_replace(summary, "\n", substitute_newline)
    # Remove space before special tokens.
    summary = tf.strings.regex_replace(summary, r" ([<\[]\S+[>\]])", b"\\1")
    summary_ids = tokenizer.tokenize(summary)
    # Add [EOS] (1) special tokens.
    suffix = tf.constant([1])
    summary_ids = tf.concat([summary_ids, suffix], axis=0)
    if isinstance(summary_ids, tf.RaggedTensor):
        summary_ids = summary_ids.to_tensor(0)
    summary_ids = summary_ids[:max_decoder_length]

    return document_ids, summary_ids



def do_masking(example):
    print(example)
    text = example["text"]
    print(text)


split = "train"
input_files = tf.io.gfile.glob(
          os.path.join(data_dir, "*{}.tfrecord*".format(split)))

d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
d = d.shuffle(buffer_size=len(input_files))

# Non deterministic mode means that the interleaving is not exact.
# This adds even more randomness to the training pipeline.
d = d.interleave(tf.data.TFRecordDataset,
                 deterministic=False,
                 num_parallel_calls=tf.data.experimental.AUTOTUNE)

for element in d.as_numpy_iterator():
  print(element.decode('euc-kr'))