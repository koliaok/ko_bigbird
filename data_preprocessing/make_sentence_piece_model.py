import sentencepiece as spm
import argparse

parse = argparse.ArgumentParser(description="input data 처리")
parse.add_argument('--input_data', type=str)
parse.add_argument('--output_model', type=str, default="./bpe_model/ko_big_bird")

added_parser = parse.parse_args()

original_define_token = "'<pad>', '</s>', '<s>', '<sep_0>', '<sep_1>', '<sep_2>', '<sep_3>', '<sep_4>', '<no_saic_raw_sp>', '<length_0>', '<length_1>', '<length_2>', '<length_3>', '<length_4>', '<length_5>', '<length_6>', '<length_7>', '<length_8>', '<length_9>', '<length_10>', '<length_11>', '<length_12>', '<length_13>', '<length_14>', '<length_15>', '<length_16>', '<length_17>', '<length_18>', '<length_19>', '<length_20>', '<length_s>', '<length_m>', '<length_l>', '<::::>', '<space>', '<unused_34>', '<unused_35>', '<unused_36>', '<no_query_sp>', '<no_passage_raw_sp>', '<unused_39>', '<unused_40>', '<unused_41>', '<unused_42>', '<unused_43>', '<unused_44>', '<unused_45>', '<unused_46>', '<unused_47>', '<unused_48>', '<unused_49>', '<paraphrasing_0>', '<paraphrasing_1>', '<paraphrasing_2>', '<paraphrasing_3>', '<paraphrasing_4>', '<paraphrasing_5>', '<paraphrasing_6>', '<paraphrasing_7>', '<paraphrasing_8>', '<empty>', '<row>', '<cn>', '<t>', '<ans>', '[CLS]', '[SEP]', '[MASK]', '<unused_67>', '<unused_68>', '<unused_69>', '<unused_70>', '<unused_71>', '<unused_72>', '<unused_73>', '<unused_74>', '<unused_75>', '<unused_76>', '<unused_77>', '<unused_78>', '<unused_79>', '<unused_80>', '<unused_81>', '<unused_82>', '<unused_83>', '<unused_84>', '<unused_85>', '<unused_86>', '<unused_87>', '<unused_88>', '<unused_89>', '<unused_90>', '<unused_91>', '<unused_92>', '<unused_93>', '<unused_94>', '<unused_95>', '<unused_96>', '<unused_97>', '<unused_98>', '<unk>',"
user_define_token = "'<sep_0>', '<sep_1>', '<sep_2>', '<sep_3>', '<sep_4>', '<no_saic_raw_sp>', '<length_0>', '<length_1>', '<length_2>', '<length_3>', '<length_4>', '<length_5>', '<length_6>', '<length_7>', '<length_8>', '<length_9>', '<length_10>', '<length_11>', '<length_12>', '<length_13>', '<length_14>', '<length_15>', '<length_16>', '<length_17>', '<length_18>', '<length_19>', '<length_20>', '<length_s>', '<length_m>', '<length_l>', '<::::>', '<space>', '<unused_34>', '<unused_35>', '<unused_36>', '<no_query_sp>', '<no_passage_raw_sp>', '<unused_39>', '<unused_40>', '<unused_41>', '<unused_42>', '<unused_43>', '<unused_44>', '<unused_45>', '<unused_46>', '<unused_47>', '<unused_48>', '<unused_49>', '<paraphrasing_0>', '<paraphrasing_1>', '<paraphrasing_2>', '<paraphrasing_3>', '<paraphrasing_4>', '<paraphrasing_5>', '<paraphrasing_6>', '<paraphrasing_7>', '<paraphrasing_8>', '<empty>', '<row>', '<cn>', '<t>', '<ans>', '[CLS]', '[SEP]', '[MASK]', '<unused_67>', '<unused_68>', '<unused_69>', '<unused_70>', '<unused_71>', '<unused_72>', '<unused_73>', '<unused_74>', '<unused_75>', '<unused_76>', '<unused_77>', '<unused_78>', '<unused_79>', '<unused_80>', '<unused_81>', '<unused_82>', '<unused_83>', '<unused_84>', '<unused_85>', '<unused_86>', '<unused_87>', '<unused_88>', '<unused_89>', '<unused_90>', '<unused_91>', '<unused_92>', '<unused_93>', '<unused_94>', '<unused_95>', '<unused_96>', '<unused_97>', '<unused_98>'"

user_define_token = user_define_token.replace("\'", '')
user_define_token = user_define_token.split(',')

original_define_token = original_define_token.replace("\'", '')
original_define_token = original_define_token.split(',')

input_file = added_parser.input_data
model_name = added_parser.output_model
vocab_size = 15000
model_type = "bpe"
max_sentence_len = 5000


class SentencePiece(object):

    def __init__(self):
        pass


    def make_sentence_piece(self):

        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_name,
            vocab_size=vocab_size,
            model_type=model_type,
            max_sentence_length=max_sentence_len,
            unk_id=100,
            bos_id=2,
            eos_id=1,
            pad_id=0,
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
            pad_piece="<pad>",
            user_defined_symbols=user_define_token)

    def load_sentece_pice_model(self):

        self.spm = spm.SentencePieceProcessor()
        self.spm.load(model_name+'.model')
        print(self.spm.get_piece_size())
        vocab_size = self.spm.get_piece_size()
        for id in range(vocab_size):
            piece = self.spm.id_to_piece(id)

            if id >= 0 and id <= 100:
                original_piece = original_define_token[id].strip()
                if piece == original_piece:
                    print('matched: ', piece)
                    continue
            print(piece)


def main():
    sentence_piecemodel = SentencePiece()
    #sentence_piecemodel.make_sentence_piece()
    sentence_piecemodel.load_sentece_pice_model()

if __name__ == '__main__':
    main()
