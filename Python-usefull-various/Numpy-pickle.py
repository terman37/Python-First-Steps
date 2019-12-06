import numpy as np
import time
import pickle
import pandas as pd
import os


def multiply_(x):
    y = x ** 2
    return y


if __name__ == '__main__':

    z = np.random.rand(5, 1)
    # print(multiply_(z))
    zz = list(z)
    # print(z.tolist())
    z_ = z.tolist()

    zz_ = [item for sub in z_ for item in sub]
    # print(zz_)

    list_ = []
    for i in zz:
        list_.append(i[0])
    # print(list_)

    # list comprehension vs for loop - much faster
    t1 = time.time()
    for i in zz:
        print(i)
    t2 = time.time()
    print((t2 - t1) * 1000)

    t3 = time.time()
    [print(i) for i in zz]
    t4 = time.time()
    print((t4 - t3) * 1000)

    a = [sub for element in zz for sub in element]
    print(a)

    # current folder
    print('current path:' + os.getcwd())

    # read write txt files.
    with open('whatever.txt', 'w') as f:
        # for i in a:
        #     f.write(str(i)+'\n')
        # or
        [f.write(str(i) + '\n') for i in a]

    with open('whatever.txt', 'r') as f:
        ad = f.read()
        print(ad)

    # read write with pickle
    with open('whatever.bin', 'wb') as f:
        pickle.dump(a, f)

    with open('whatever.bin', 'rb') as f:
        ad = pickle.load(f)
        print(ad)
