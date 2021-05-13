import tensorflow as tf
from bigbird.create_dataset.tokenization import FullTokenizer

class TextUtil():

    def __init__(self, vocab_model):
        self.tokenizer = FullTokenizer(vocab_file=vocab_model) # bpe 기반 vocab


    def text_to_tf_ids(self, text):

        tf_text_ids = self.tokenizer.tf_tokenizer.tokenize(text)

        if isinstance(tf_text_ids, tf.RaggedTensor):
            ids = tf_text_ids.to_tensor(0)

        res = self.tokenizer.tf_tokenizer.id_to_string(tf_text_ids)
        tf_res_data_list = [text.decode('UTF-8') for text in res.numpy()]

        return tf_text_ids, tf_res_data_list


    def original_text_data_matching_bpe_data(self, text, tf_text_ids):

        data_ids = tf_text_ids.numpy().tolist()
        data_list = {}
        word_index_list = []
        index_list = []
        cnt = 0
        text_list = text.split()
        first_flag = False

        for i, idx in enumerate(data_ids):
            res_data = self.tokenizer.inv_vocab[idx]

            data_list[i] = res_data

            word_index_list.append(i)

            if res_data[0] == "▁" and first_flag or i == len(data_ids) - 1:
                index_list.append((text_list[cnt], word_index_list[:-1]))
                cnt += 1
                word_index_list = [word_index_list[-1]]
                first_flag = False

                if i == len(data_ids) - 1 and res_data[0] == "▁":
                    index_list.append((text_list[cnt], word_index_list))

            if res_data[0] == "▁" and not first_flag:
                first_flag = True

        return index_list


    def text_to_tfdata(self, text):
        return tf.constant(text, dtype=tf.string)




