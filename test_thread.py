# import multiprocessing

# def f(x):
#     return x * x

# cores = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(processes=cores)
# xs = range(5)

# # method 1: map
# print(pool.map(f, xs))  # prints [0, 1, 4, 9, 16]

# # method 2: imap
# for y in pool.imap(f, xs):
#     print(y)            # 0, 1, 4, 9, 16, respectively

#  y in pool.imap_unordered(f, xs):
#     print(y)           # may be in any order
from multiprocessing import Pool
import time

COUNT = 50000000
def countdown(n):
    while n>0:
        n -= 1

if __name__ == '__main__':
    pool = Pool(processes=2)
    start = time.time()
    r1 = pool.apply_async(countdown, [COUNT//2])
    r2 = pool.apply_async(countdown, [COUNT//2])
    pool.close()
    pool.join()
    end = time.time()
    print('Time taken in seconds -', end - start)