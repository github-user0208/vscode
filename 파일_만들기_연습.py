f=open('newText.txt','w')
for i in range(1,11):
    data=f'{i}st line.\n'
    f.write(data)
f.close()