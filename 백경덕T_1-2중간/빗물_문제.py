def rain(heights):
    l = heights
    for i in range(1, len(l)-1):
        m = min(max(l[:i], l[i+1:]))

heights = list(map(int, input().split()))
rain(heights)