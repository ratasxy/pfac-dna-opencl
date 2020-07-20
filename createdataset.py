from random import seed
from random import randint
import pprint


def randomstring(size=32):
    di = ['A', 'C', 'G', 'T']
    word = ''
    for i in range(0,32):
        word = word + di[randint(0, 3)]
    return word

def randomdataset(datasize=10, size=32):
    dataset = []
    for i in range(0, datasize):
        dataset.append(randomstring(size))

    return dataset

seed(154534543654643)

n = 3000000
z = 128
t = 'pattern'
t = 'test'

dataset = randomdataset(n, z)

with open('./datasets/%s-%i-%i.txt' % (t, n, z), 'w') as f:
    for item in dataset:
        f.write("%s\n" % item)