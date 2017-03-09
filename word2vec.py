# word2vec 

import gensim

sentences = [['first', 'sentence'], ['second', 'sentence']]

model = gensim.models.Word2Vec(sentences, min_count=1)

print(model)
#for vec in model:
#	print(vec)

