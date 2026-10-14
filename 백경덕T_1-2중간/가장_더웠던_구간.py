n = int(input())
t = list(map(int, input().split()))
cur = t[0]
dp = [0]*n
for i in range(n):
    dp[i] = max(dp[i-1]+t[i], t[i])
print(max(dp))