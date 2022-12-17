import gensim.downloader
from numpy.linalg import norm
import numpy as numpy

glove_vectors = gensim.downloader.load( 'word2vec-google-news-300')
print(glove_vectors["hey"])