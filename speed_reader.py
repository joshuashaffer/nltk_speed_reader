#!/usr/bin/python3
import sys
import re
import signal
import itertools
from functools import namedtuple

import climate
from PyQt5.QtWidgets import QApplication, QLabel
from PyQt5 import QtCore, QtWidgets
import nltk
import pyphen

Labeled_Word = namedtuple('Labeled_Word', ['text', 'part_of_speech'])
Split_Word = namedtuple('Split_Word', ['start', 'mid', 'end'])

event_timer = QtCore.QTimer()
sentence_iter = None
word_iter = None

word_width = 28
mid_width = word_width // 2
hypenizer = pyphen.Pyphen(lang='en_US')


def iterate_sentences(file):
    with open(file, 'r') as fid:
        raw_test = fid.read()
    tokenizer = nltk.TweetTokenizer(reduce_len=True)
    sentences = nltk.sent_tokenize(raw_test)
    for sentence in sentences:
        s = re.sub("[^0-9a-zA-Z \n-?!.:\u2019]+", '', sentence).strip()
        yield map(lambda x: Labeled_Word(*x), nltk.pos_tag(tokenizer.tokenize(s)))


def hypenize(word_particle_list):
    for word_particle in word_particle_list:
        if len(word_particle[0]) > word_width:
            temp_word = word_particle.text
            # Split closest to the middle of the word
            idx_split = min(map(lambda x: (x[0], abs((len(temp_word) / 2.0 - x[1])), x[1]),
                                enumerate(hypenizer.positions(temp_word))),
                            key=lambda x: x[1])[2]
            yield Labeled_Word(temp_word[:idx_split] + '-', word_particle.part_of_speech)
            yield Labeled_Word(temp_word[idx_split:], word_particle.part_of_speech)
        else:
            yield word_particle


def wpm_to_ms(words_per_min):
    return 1.0 / ((words_per_min / 60.0) * (1.0 / 1000.0))


def split_word(word):
    mid = len(word) // 2
    split_point = mid
    for idx in range(mid, mid // 2, -1):
        if word[idx] in list('aeuioöäüAEUIOÖÄÜ'):
            split_point = idx
            break
    word_start = str(word[:split_point]).rjust(mid_width).replace(' ', '&nbsp;')
    word_mid = word[split_point]
    word_end = str(word[split_point + 1:]).ljust(mid_width - 1).replace(' ', '&nbsp;')
    return Split_Word(word_start, word_mid, word_end)


def update_label(label_start, default_delay, noun_delay_multiplier, verb_delay_multiplier):
    word = next_word()
    if word is None:
        sys.exit(0)

    color_start = 'black'
    color_mid = '#ae01ae'
    color_end = 'black'
    label_template_start = """
    <span style='font-size:24px;font-size:20vw; font-weight:600; color:{}; font-family: monospace;'>{{}}</span>
        """.format(color_start)
    label_template_mid = """
    <span style='font-size:24px;font-size:20vw; font-weight:600; color:{};font-family: monospace;'>{{}}</span>""".format(
        color_mid)
    label_template_end = """
    <span style='font-size:24px;font-size:20vw; font-weight:600; color:{};font-family: monospace;'>{{}}</span> """.format(
        color_end)
    rword = re.sub('[^0-9a-zA-Z-]+', '', word.text).strip().lower()
    if rword == '':
        # Punctuation mark
        event_timer.setInterval(default_delay * 1.5)
        return

    next_delay = default_delay
    if len(rword) > 6:
        next_delay = (default_delay / 2.0) * (len(rword) - 6)

    word_part = word.part_of_speech
    if word_part == 'NN' or word_part == 'NNP' or word_part == 'PRP':
        # Nouns or preps.
        rword = rword.capitalize()
        next_delay *= noun_delay_multiplier
    elif word_part == 'VB':
        # main verbs
        rword = rword.upper()
        next_delay *= verb_delay_multiplier

    sword = split_word(rword)
    label_start.setText(label_template_start.format(sword.start) +
                        label_template_mid.format(sword.mid) +
                        label_template_end.format(sword.end))
    event_timer.setInterval(int(next_delay))


def next_word():
    global sentence_iter
    global word_iter
    word = None
    try:
        word = next(word_iter)
    except:
        try:
            word_iter = hypenize(next(sentence_iter))
            word = next(word_iter)
        except:
            pass
    return word


@climate.annotate(
    text_file=('Ascii input', 'positional', None, str),
    wpm=('Words per minute', 'positional', None, int),
    noun_delay=('Noun delay multiplier', 'option', None, float),
    verb_delay=('Verb delay multiplier', 'option', None, float)
)
def speed_reader_main(text_file, wpm, noun_delay=2.0, verb_delay=2.0):
    global sentence_iter
    global word_iter

    sentence_iter, sentence_iter_orig = itertools.tee(iterate_sentences(text_file))
    sentence_iter_orig, sentence_count_iter = itertools.tee(sentence_iter_orig)
    # sentence_count = sum(sum(b[1] == 'VB' for b in a) for a in sentence_count_iter)
    # print("{} Sentences".format(sentence_coun))
    word_iter = hypenize(next(sentence_iter))

    app = QApplication(sys.argv)
    window = QtWidgets.QMainWindow()
    lmainFirst = QLabel()
    default_delay = wpm_to_ms(wpm)
    event_timer.timeout.connect(lambda: update_label(lmainFirst, default_delay, noun_delay, verb_delay))
    event_timer.start(1000)

    window.setCentralWidget(lmainFirst)
    window.setMinimumWidth(500)
    window.show()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    sys.exit(app.exec_())


if __name__ == '__main__':
    climate.call(speed_reader_main)
    # r =Labeled_Word(*('ok','VB'))
    # IPython.embed()
