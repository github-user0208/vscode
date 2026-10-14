from itertools import product
from fractions import Fraction
import mpmath as mp
mp.mp.dps = 50

def compute(n):
    denoms = [2*k+1 for k in range(n)]
    C = 1
    for d in denoms:
        C *= d
    total_g1 = mp.mpc(0)
    for signs in product([1,-1], repeat=n):
        lam = sum(Fraction(s,1)/Fraction(d) for s,d in zip(signs, denoms))
        lam_mp = mp.mpf(lam.numerator)/mp.mpf(lam.denominator)
        num_minus = signs.count(-1)
        sign_coef = (-1)**num_minus
        term = sign_coef * (mp.mpc(0,1)*lam_mp)**(n-1) / mp.factorial(n-1)
        if lam_mp > 0:
            total_g1 += term
    Cfull = mp.mpf(C) / ((mp.mpc(0,2))**n)
    Res_g1 = Cfull * total_g1
    return 2*mp.pi*mp.mpc(0,1)*Res_g1

pi = mp.pi
for n in range(1,12):
    val = compute(n)
    print(f"n={n}: value={mp.nstr(val.real,20)}  diff={mp.nstr(val.real-pi,10)}")