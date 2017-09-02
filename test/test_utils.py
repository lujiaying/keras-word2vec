import os
import random

from word2vec import utils

class TestUtils(object):

    def setUp(self):
        self._fake_corpus_file = './test/fake_corpus'
        self._gen_fake_corpus_file(self._fake_corpus_file)

    def tearDown(self):
        os.system('rm %s' % (self._fake_corpus_file))

    def test_build_word_dict_by_file(self):
        word_id_dict, id_word_dict, word_counter = utils.build_word_dict_by_file(self._fake_corpus_file)

        assert len(word_id_dict) == 29
        assert word_id_dict['and'] == 0
        assert id_word_dict[0] == 'and'
        assert word_counter['and'] == 2
        assert word_id_dict['<unk>'] == 28
        assert word_counter['<unk>'] == 0

    def test_build_word_dict_by_file_min_word_freq(self):
        word_id_dict, id_word_dict, word_counter = utils.build_word_dict_by_file(self._fake_corpus_file, min_word_freq=2)

        assert len(word_id_dict) == 5
        assert word_id_dict['and'] == 0
        assert id_word_dict[0] == 'and'
        assert word_counter['and'] == 2
        assert word_id_dict['<unk>'] == 4
        assert word_counter['<unk>'] == 24

    def test_build_negative_sample_array(self):
        word_id_dict, id_word_dict, word_counter = utils.build_word_dict_by_file(self._fake_corpus_file, min_word_freq=2)
        negative_sample_array = utils.build_negative_sample_array(word_counter, word_id_dict, 30)

        assert len(word_id_dict) == 5
        assert word_id_dict['and'] == 0
        assert id_word_dict[0] == 'and'
        assert word_counter['and'] == 2
        assert word_id_dict['<unk>'] == 4
        assert word_counter['<unk>'] == 24
        assert negative_sample_array[0] == 0
        assert negative_sample_array[-1] == 4

    def test_generate_negative_sample(self):
        word_id_dict, id_word_dict, word_counter = utils.build_word_dict_by_file(self._fake_corpus_file, min_word_freq=2)
        negative_sample_array = utils.build_negative_sample_array(word_counter, word_id_dict, 30)
        random.seed(10)
        negative_sample = utils.generate_negative_sample(negative_sample_array)

        assert negative_sample == 4

    def test_NEG_reader_creator(self):
        word_id_dict, id_word_dict, word_counter = utils.build_word_dict_by_file(self._fake_corpus_file, min_word_freq=2)
        negative_sample_array = utils.build_negative_sample_array(word_counter, word_id_dict, 30)
        reader = utils.NEG_reader_creator(word_id_dict, self._fake_corpus_file, negative_sample_array, word_counter)()

        reader.next()
        reader.next()
        reader.next()
        reader.next()


    def _gen_fake_corpus_file(self, filename):
        with open(filename, 'w') as fwrite:
            fwrite.write('a neural probabilistic language model\n')
            fwrite.write('distributed representations of words and phrases and their compositionality\n')
            fwrite.write('bag of tricks for efficient text classification\n')
            fwrite.write('enriching word vectors with subword information\n')
            fwrite.write('fasttext.zip compressing text classification models\n')
