"""
Papers:
    [1] Mikolov T, Chen K, Corrado G, et al. Efficient Estimation of Word Representations in Vector Space[J]. Computer Science, 2013. (https://arxiv.org/pdf/1301.3781.pdf)
    [2] Mikolov T, Sutskever I, Chen K, et al. Distributed Representations of Words and Phrases and their Compositionality[J]. Advances in Neural Information Processing Systems, 2013, 26:3111-3119. (https://arxiv.org/pdf/1310.4546.pdf)
"""

from __future__ import division
import sys
import datetime
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Embedding, dot, Reshape, Dense, Merge, Activation

import utils

class SkipgramNEG(object):
    """
    Skipgram model with negative sampling to speed up.

    Methods:
        train():
        train_by_file():
        save_as_w2v_format():
    """
    def __init__(self, corpus_file, embedding_dim, min_word_freq=3, 
            context_window=2, seperator=' ', negative_sample_rate=1):
        """
        Args:
            context_window: int, furthest distance from target word
            corpus_file: string, corpus to train word embedding
            embedding_dim: int, word embedding dimension
            negative_sample_rate: int, negative_sample for each train sample = positive_sample_cnt * negative_sample_rate
        """
        self._corpus_file = corpus_file
        self._embedding_dim = embedding_dim
        self._seperator = seperator
        self._min_word_freq = min_word_freq
        self._context_window = context_window
        self._negative_sample_rate = negative_sample_rate

        self._word_id_dict, self._id_word_dict, self._word_counter = utils.build_word_dict_by_file(corpus_file, 
                seperator, min_word_freq)
        self._vocab_size = len(self._word_id_dict)
        self._model = self._init_model(self._vocab_size, embedding_dim)

    def _init_model(self, vocab_size, embedding_dim=100, ):
        # Functional paradigm
        target = Input(shape=(1,), name='target')
        context = Input(shape=(1,), name='context')
        shared_embedding = Embedding(vocab_size, embedding_dim, input_length=1, name='shared_embedding')
        embedding_target = shared_embedding(target)
        embedding_context = shared_embedding(context)
        merged_vector = dot([embedding_target, embedding_context], axes=-1)
        reshaped_vector = Reshape((1,), input_shape=(1,1))(merged_vector)
        prediction = Dense(1, input_shape=(1,), activation='sigmoid')(reshaped_vector)

        model = Model(inputs=[target, context], outputs=prediction)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        return model

    def train(self, epochs=5, batch_size=512, shuffle=True):
        negative_sample_array_size = 100000000  # due to paper [1]
        negative_sample_array = utils.build_negative_sample_array(self._word_counter, self._word_id_dict,
                negative_sample_array_size)

        for epoch_id in xrange(epochs):
            # train by batch
            batch_id = 0
            x_batch = [[],[]]
            y_batch = []
            loss_list = []
            reader = utils.NEG_reader_creator(self._word_id_dict, 
                    self._corpus_file, negative_sample_array, self._word_counter,
                    self._context_window, self._negative_sample_rate, self._seperator)
            if shuffle:
                reader = utils.shuffle(reader, batch_size*30)
            for word_ids, label in reader():
                batch_id += 1
                x_batch[0].append(word_ids[0])
                x_batch[1].append(word_ids[1])
                y_batch.append(label)
                if batch_id % (batch_size*100) == 0:
                    sys.stdout.write('\r[epoch #%d] batch #%d, train loss:%s' % (epoch_id, 
                        batch_id, np.mean(loss_list)))
                    sys.stdout.flush()
                    loss_list = []
                if batch_id % batch_size == 0:
                    X = [np.array(x_batch[0]), np.array(x_batch[1])]
                    loss = self._model.train_on_batch(X, np.array(y_batch))
                    loss_list.append(loss)
                    x_batch = [[],[]]
                    y_batch = []
            sys.stdout.write('\n%s [epoch #%d] done\n' % (
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), epoch_id))

    def save_as_w2v_format(self, output_file, use_word_id_key=False):
        """
        Store word embedding result to text file.

        Args:
            output_file: string, w2v file output path
            use_word_id_key: bool, if true, key is word_id; other wise key is word
        Returns:
            int, 0 success, else fail
        """
        with open(output_file, 'w') as fwrite:
            fwrite.write('%d %d\n' % (len(self._word_id_dict), self._embedding_dim))
            for idx, vec in enumerate(self._model.layers[2].get_weights()[0].tolist()):
                if use_word_id_key:
                    fwrite.write('%d %s\n' % (idx, ' '.join([str(_) for _ in vec])))
                else:
                    fwrite.write('%s %s\n' % (self._id_word_dict[idx], ' '.join([str(_) for _ in vec])))
