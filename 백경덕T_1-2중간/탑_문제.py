def tower(heights):
    l = heights
    ans = list()
    for i in range(len(l)):
        if i==0 or l[i]==0:
            ans.append(0)
        else:
            stack = l[:i]
            while(stack):
                if stack[-1] >= l[i]:
                    ans.append(len(stack))
                    break
                else:
                  stack.pop()
        if len(ans) != i+1:
                ans.append(0)
    return ans

n = int(input())
heights = list(map(int, input().split()))
nums = tower(heights)
for column in nums:
    print(column, end = ' ')