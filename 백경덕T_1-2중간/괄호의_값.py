# Write your solution here
def cal(s):
  Dict = {'(' : 2, '[' : 3}
  match = {')' : '(', ']' : '['}
  stack = list()
  val = 0
  for i in s:
    if i in Dict: # 여는 괄호일 경우 현재까지의 값을 stack에 저장하고 값 초기화
      stack.append((i, val))
      val = 0
    else: # 닫는 괄호일 경우, 
      if not stack or stack[-1][0] != match[i]:
        return "None"
      Open, cur = stack.pop()
      if val == 0: # 바로 앞이 여는 괄호일 경우
        num = Dict[Open]
      else:
        num = Dict[Open] * val
      val = cur + num

  if stack:
     return "None"
  return val
  
s = input()
print(cal(s))