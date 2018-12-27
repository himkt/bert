# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import six
import sys
import tensorflow as tf
import sentencepiece as spm


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token, _ = token.split('\t')
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_tokens_to_ids(vocab, tokens):
  """Converts a sequence of tokens into ids using the vocab."""
  ids = []
  for token in tokens:
      if token in vocab:
        ids.append(vocab[token])
      else:
        ids.append(vocab['<unk>'])
  return ids


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a peice of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, model_file, vocab_file, do_lower_case=True):
    self.tokenizer = SentencePieceTokenizer(model_file)
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}

  def tokenize(self, text):
    split_tokens = self.tokenizer.tokenize(text)
    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_tokens_to_ids(self.vocab, tokens)


class SentencePieceTokenizer(object):
    """Runs SentencePiece tokenization (from raw text to tokens list)"""

    def __init__(self, model_file=None, do_lower_case=True):
        """Constructs a SentencePieceTokenizer."""
        self.tokenizer = spm.SentencePieceProcessor()
        if self.tokenizer.Load(model_file):
            print("Loaded a trained SentencePiece model.")
        else:
            print("You have to set the path to a trained SentencePiece model.")
            sys.exit(1)
        self.do_lower_case = do_lower_case

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = convert_to_unicode(text)
        output_tokens = self.tokenizer.EncodeAsPieces(text)
        return output_tokens
