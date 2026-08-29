def rain():
    l=list(map(int, input().split()))
    m=max(l)
    n=len(l)
    a=[[0]*(n) for _ in range(m)]
    for i in range(n):
        for j in range(m):
            if l[i]>j:
                a[j][i]=9
    for i in range(n):
        for j in range(m):
            if a[i][j]==9:
                continue
            else:
                
    print(a)

rain()