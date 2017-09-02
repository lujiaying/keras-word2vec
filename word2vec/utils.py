"""
Papers:
    [1] Mikolov T, Chen K, Corrado G, et al. Efficient Estimation of Word Representations in Vector Space[J]. Computer Science, 2013. (https://arxiv.org/pdf/1301.3781.pdf)
    [2] Mikolov T, Sutskever I, Chen K, et al. Distributed Representations of Words and Phrases and their Compositionality[J]. Advances in Neural Information Processing Systems, 2013, 26:3111-3119. (https://arxiv.org/pdf/1310.4546.pdf)
"""

from __future__ import division
import random

def build_word_dict_by_file(corpus_file, seperator=' ', min_word_freq=0):
    """
    Args:
        corpus_file: string
        seperator: string, to seperate word in each line of *corpus_file*
        min_word_freq: int
    Returns:
        word_id_dict: dict
        id_word_dict: dict
        word_counter: dict, {word1: counter1(3000)}
    """
    word_counter = {}
    total_cnt = 0
    with open(corpus_file) as fopen:
        for line in fopen:
            for word in line.strip().split(seperator):
                if word not in word_counter:
                    word_counter[word] = 0
                word_counter[word] += 1
                total_cnt += 1
    word_freq = filter(lambda _:_[1] >= min_word_freq, word_counter.iteritems())
    word_counter_sorted = sorted(word_freq, key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*word_counter_sorted))
    word_id_dict = dict(zip(words, range(len(words))))
    word_id_dict['<unk>'] = len(words)
    id_word_dict = dict([(_[1], _[0]) for _ in word_id_dict.iteritems()])

    word_freq_cnt = sum([_[1] for _ in word_freq])
    word_counter = dict(word_freq)
    word_counter['<unk>'] = total_cnt - word_freq_cnt

    assert len(id_word_dict) == len(word_id_dict)
    print('Number of words: %d' % (len(word_id_dict)))
    return word_id_dict, id_word_dict, word_counter

def build_negative_sample_array(word_counter, word_id_dict, array_size=100000000):
    """
    Element of negative sample array is word_id, and count(word_i) = P(word_i) * array_size.
    Due to [1], P(word_i) = unigram(word_i)^3/4 / sum(unigram(word_j)^3/4)

    Args:
        word_counter: dict, {word1: freq1, ...}
        word_id_dict: dict, {word1: word_id1, ...}
        array_size: int, negative_sample_array size
    Returns:
        negative_sample_array: list
    """
    negative_sample_array = [0 for _ in xrange(array_size)]
    idx = 0

    func_freq = dict([(_[0], _[1]**0.75) for _ in word_counter.iteritems()])
    sum_freq = sum(func_freq.itervalues())
    for word, freq in func_freq.iteritems():
        if idx >= array_size:
            break
        word_sample_num = int(round(freq / sum_freq * array_size))
        word_id = word_id_dict[word]
        for _ in xrange(word_sample_num):
            if idx >= array_size:
                break
            negative_sample_array[idx] = word_id
            idx += 1
    return negative_sample_array

def generate_negative_sample(negative_sample_array):
    idx = random.randrange(0, len(negative_sample_array))
    return negative_sample_array[idx]

def NEG_reader_creator(word_id_dict, corpus_file, negative_sample_array, word_counter,
        context_window=2, negative_sample_rate=1, seperator=' ',
        subsample_frequent_words=True, random_context_window=True):
    """
    Basic NEG reader creator, generate <target, context> and <target, noise>
    Context window is fixed. High frequency word is not subsampled.

    Args:
        word_id_dict: dict
        corpus_file: string
        negative_sample_array: list
        context_window: int, furthest distance from target word
        negative_sample_rate: int, negative_sample for each train sample = positive_sample_cnt * negative_sample_rate
        seperator: string
        subsample_frequent_words: bool, due to [2], each input word is discarded with prob: P(w_i) = 1 - sqrt(t/f(w_i))
        random_context_window: bool, dut to [1], random context window = randrange(1, context_window)
    """
    def reader():
        t = 10 ** -5
        total_cnt = sum(word_counter.itervalues())
        word_frequency_dict = dict([(_[0], _[1]/total_cnt) for _ in word_counter.iteritems()])
        with open(corpus_file) as fopen:
            for line in fopen:
                if random_context_window:
                    cur_context_window = random.randrange(1, context_window+1)
                else:
                    cur_context_window = context_window
                line_list = line.strip().split(seperator)
                word_ids = [word_id_dict.get(_, word_id_dict['<unk>']) for _ in line_list]
                for i in xrange(len(word_ids)):
                    target = word_ids[i]
                    target_word = line_list[i] if line_list[i] in word_id_dict else '<unk>'
                    if subsample_frequent_words:
                        if is_discard_word(word_frequency_dict[target_word], t):
                            continue
                    # generate positive sample
                    context_list = []
                    j = i - cur_context_window
                    while j <= i + cur_context_window and j < len(word_ids):
                        if j >= 0 and j != i:
                            context_list.append(word_ids[j])
                            yield ((target, word_ids[j]), 1)
                        j += 1
                    # generate negative sample
                    for _ in xrange(len(context_list)*negative_sample_rate):
                        ne_idx = generate_negative_sample(negative_sample_array)
                        while ne_idx in context_list:
                            ne_idx = generate_negative_sample(negative_sample_array)
                        yield ((target, ne_idx), 0)
    return reader

def shuffle(reader, buf_size):
    """
    Creates a data reader whose data output is shuffled.

    Output from the iterator that created by original reader will be
    buffered into shuffle buffer, and then shuffled. The size of shuffle buffer
    is determined by argument buf_size.

    Args:
        reader: callable, the original reader whose output will be shuffled.
        buf_size, int, shuffle buffer size.
    Returns:
        data_reader: callable, the new reader whose output is shuffled.
    """
    import random
    def data_reader():
        buf = []
        for e in reader():
            buf.append(e)
            if len(buf) >= buf_size:
                random.shuffle(buf)
                for b in buf:
                    yield b
                buf = []

        if len(buf) > 0:
            random.shuffle(buf)
            for b in buf:
                yield b

    return data_reader

def is_discard_word(frequency, t):
    p = 1 - (t / frequency) ** 0.5
    if random.random() <= p:
        return True
    else:
        return False
