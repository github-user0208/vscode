import numpy as np
import matplotlib.pyplot as plt

# 시스템 파라미터
a = 1.1
b = 0.5
k = 1.8
r = 0.1

# 초기값
x0 = 10
N = 30

# 리스트 초기화
x = [x0]
u = []
J = [0]

# 시뮬레이션
for n in range(N):
    u_n = -k * x[n]
    x_next = a * x[n] + b * u_n
    
    x.append(x_next)
    u.append(u_n)
    
    cost = x[n]**2 + r * u_n**2
    J.append(J[n] + cost)

# 그래프
plt.figure()

plt.plot(x, label="State x_n")
plt.plot(u, label="Control u_n")
plt.plot(J, label="Cost J_n")

plt.legend()
plt.xlabel("n")
plt.title("Rocket Attitude Control via Sequence Convergence")
plt.show()