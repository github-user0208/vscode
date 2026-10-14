def rain(heights):
    l = heights
    ans = list()
    for i in range(1, len(l)-1):
        m = min(max(l[:i+1]), max(l[i:])) - l[i]
        ans.append(m)

    return sum(ans)

h, w = input().split()
heights = list(map(int, input().split()))
print(rain(heights))