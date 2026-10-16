# 두번째 풀이, 최적화 적용
def tower(n, heights):
    ans = [0]*n
    stack = []                      # 후보 탑의 인덱스 (높이가 위로 갈수록 작아짐)
    for i in range(n):
        while stack and heights[stack[-1]] < heights[i]:
            stack.pop()             # 낮은 탑은 영구 제거
        if stack:
            ans[i] = stack[-1] + 1  # 가장 가까운 후보, 1-indexed
        stack.append(i)
    return ans

n = int(input())
height = list(map(int, input().split()))
nums = tower(n, height)
for columns in nums:
    print(columns, end = ' ')

# 처음 풀이, 최적화 안됨
'''
def tower(n, heights):
    l = heights
    ans = list()
    for i in range(n):
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
nums = tower(n, heights)
for column in nums:
    print(column, end = ' ')
'''