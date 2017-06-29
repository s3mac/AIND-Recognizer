a=[]

x=[1,2,3,4]

for i in x:
    b=[]
    for c in x:
        b.append(c)
    print(b)
    a.append(b)

print(a)
