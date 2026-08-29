n=int(input())
m=int(input())
list=[[] for _ in  range(n+1)]

for i in range(m):
    a,b=map(int, input().split())
    list[a].append(b)
    list[b].append(a)

virus=[0]*(n+1)
virus[1]=1

while True:
    check=0
    for i in range(1,n+1):
        if virus[i]==1:
            for j in list[i]:
                if virus[j]==0:
                    virus[j]=1
                    check=1
    if check==0:
        break

print(sum(virus)-1)