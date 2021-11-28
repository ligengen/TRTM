import matplotlib.pyplot as plt
import os
import re
import pdb

file = '/Users/ligen/Desktop/auglog.txt'

def process(file):
  with open(file, 'r') as f:
    data = f.read().split('\n')
  f.close()

  lis = []
  for i in data:
    if '***** Epoch' in i:
      lis.append(i)

  train = []
  eval = []
  for idx, i in enumerate(lis):
    if idx % 2 == 0:
      train.append(float(i.split(' ')[-2]))
    else:
      eval.append(float(i.split(' ')[-2]))
  return train, eval

if __name__ == '__main__':
    train, eval = process(file)
    train2, eval2 = process('/Users/ligen/Desktop/50log.txt')
    train3, eval3 = process('/Users/ligen/Desktop/152log.txt')
    plt.clf()
    # plt.plot([i for i in range(len(train2))], train2, label='train50', linestyle='dashed')
    plt.plot([i for i in range(25)], train2[:25], label='train50', linestyle='dashed')
    # plt.plot([i for i in range(len(eval2))], eval2, label='eval50')
    plt.plot([i for i in range(25)], eval2[:25], label='eval50')
    # plt.plot([i for i in range(len(train))], train, label='train50-aug', linestyle='dashed')
    # plt.plot([i for i in range(len(eval))], eval, label='eval50-aug')
    # plt.plot([i for i in range(len(train3))], train3, label='train152', linestyle='dashed')
    # plt.plot([i for i in range(len(eval3))], eval3, label='eval152')
    plt.plot([i for i in range(16)], [0.11274, 0.074692, 0.06344, 0.056911, 0.048233, 0.041944, 0.039632, 0.038465, 0.035395, 0.034913, 0.034066, 0.032339, 0.031525, 0.030952, 0.030109, 0.030381], label='train image loss', linestyle='dashed')
    plt.plot([i for i in range(16)], [0.09744, 0.07015, 0.065068, 0.055526, 0.048550, 0.048837, 0.040914, 0.043306, 0.038622, 0.037935, 0.034907, 0.036151, 0.032617, 0.032332, 0.031121, 0.028129], label='eval image loss')
    plt.legend()
    plt.show()
    plt.close()


