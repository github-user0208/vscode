def tower(height):
    l = height
    

n = int(input())
height = list(map(int, input().split()))
nums = tower(height)
for column in nums:
    print(column, end = ' ')