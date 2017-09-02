import os
import time

from word2vec import skipgram

class TestUtils(object):

    def setUp(self):
        self._fake_corpus_file = './test/text8'
        self._output_w2v_file = './test/text8_w2v.vec'
        #self._maybe_download(self._fake_corpus_file, 31344016)

    def tearDown(self):
        pass
        """
        os.system('rm %s' % (self._fake_corpus_file))
        os.system('rm %s' % (self._output_w2v_file))
        """

    def test_train_SkipgramNEG(self):
        model = skipgram.SkipgramNEG(self._fake_corpus_file, 100, min_word_freq=10)
        model.train(epochs=2, batch_size=1024)
        model.save_as_w2v_format(self._output_w2v_file)

    def _maybe_download(self, filename, expected_bytes):
        """Download a file if not present, and make sure it's the right size."""
        from six.moves import urllib

        url = 'http://mattmahoney.net/dc/'
        if not os.path.exists(filename):
          filename, _ = urllib.request.urlretrieve(url + filename, filename)
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
          print('Found and verified', filename)
        else:
          print(statinfo.st_size)
          raise Exception(
              'Failed to verify ' + filename + '. Can you get to it with a browser?')
        return filename
