import sys
sys.setrecursionlimit(2*(10**4))

a = [0] * (2 * (10**4))
a[1] = 1
a[2] = 2

def f(n):
    if n == 1: return 1
    elif n == 2: return 2

    if a[n] != 0:
        return a[n]
    else:
        a[n] = f(n-1) + f(n-2)
        return a[n]

m = int(input())
print(f(m)%100000007)