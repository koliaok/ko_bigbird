import tensorflow as tf
import numpy as np
import os
import argparse

from scipy.special import softmax
from bigbird.create_dataset.text_util import TextUtil


tf.compat.v2.enable_v2_behavior()

parser = argparse.ArgumentParser(description="serving parameter")
parser.add_argument('--serve_model_dir', type=str, help='insert serve model directory', required=True)
parser.add_argument('--vocab_dir', type=str, help='insert vocab directory', required=True)
parser.add_argument('--test_text', type=str, help='insert vocab directory', required=True)
args = parser.parse_args()


class ServingManager(object):
    """
    Serving model을 로드, 테스트, 관리 하는 Class
    """

    def __init__(self):
        # Text Tokenize, De-Tokenize, Tokenize 된 텍스트를 원래 텍스트로 변환하고 그 Index를 관리하는 Class
        self.text_util = TextUtil(vocab_model=args.vocab_dir)


    def serving_model_load(self, model_dir):
        """
        Serving Model Load
        """
        original_path = os.listdir(model_dir)[0]
        model_path = model_dir + original_path
        serv_model = tf.saved_model.load(model_path, tags='serve')
        default_serve_model = serv_model.signatures['serving_default']

        return default_serve_model


    def attention_test(self):
        """
        불러온 Serving Model이 텍스트 어텐션을 어떻게 수행하는지 test
        """

        print("original text: ", args.test_text)

        tf_tokenized_ids, tokenized_text = self.text_util.text_to_tf_ids(args.test_text)

        print("original tokenized text:", ' '.join(tokenized_text))

        tf_data = self.text_util.text_to_tfdata(args.test_text)

        original_text_matched_tokenized_data = self.text_util.original_text_data_matching_bpe_data(args.test_text, tf_tokenized_ids)

        print("original text matched tokenized format ", original_text_matched_tokenized_data)

        serve_model = self.serving_model_load(args.serve_model_dir)

        id_to_original_text = {i: text for i, text in enumerate(args.test_text.split())}

        self.attention_original_word(serve_model=serve_model, string_tensor=tf_data,
                                     matching_tokenize_text=original_text_matched_tokenized_data,
                                     tokenized_text=tokenized_text,
                                     id_to_original_text=id_to_original_text)


    def attention_original_word(self, serve_model, string_tensor,
                                matching_tokenize_text,
                                tokenized_text, id_to_original_text):

        """
        불러온 모델을 텍스트에 대한 어텐션을 수행하는 함수
        """

        predicted_summary = serve_model(string_tensor)
        seq_embedding = predicted_summary['seq-embeddings'][0]
        res = seq_embedding.numpy()
        res = res[:len(tokenized_text)]
        res = np.average(res, axis=-1)
        res_numpy = []

        # 원본 텍스트 형식으로 만든 단어별로 Representation하기 위한 과정
        for (v, idx_list) in matching_tokenize_text:
            res_v = res[idx_list[0]:idx_list[-1] + 1]
            res_numpy.append(np.average(res_v)) # 원본텍스트에 대한 단어의 representation을 평균
        res = softmax(res_numpy, axis=-1) # 가장 의미가 큰 것을 구하기 위해서 Softmax

        index_max = {}
        for i, data in enumerate(res):
            index_max[i] = data

        # 가장 attention이 큰 값 부터 정렬
        res_sorted = sorted(index_max.items(), key=lambda item: item[1], reverse=True)

        print("attention text from original text")

        for i, (idx, v) in enumerate(res_sorted):
            print('Text: ', id_to_original_text[idx], '  Rank: ', i, '    Score: ', v)



serving_manager = ServingManager()
serving_manager.attention_test()



