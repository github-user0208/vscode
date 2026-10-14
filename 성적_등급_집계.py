n = int(input())
Dict = dict()
for _ in range(n):
  a,b = input().split()
  b=int(b)
  if b>=90:
    Dict[a] = 'A'
  elif b>=80:
    Dict[a] = 'B'
  elif b>=70:
    Dict[a] = 'C'
  elif b>=60:
    Dict[a] = 'D'
  else:
    Dict[a] = 'F'

m = int(input())
for i in range(m):
  a = input()
  cnt = 0
  for j in Dict.values():
    if j == a:
      cnt += 1
  print(cnt)