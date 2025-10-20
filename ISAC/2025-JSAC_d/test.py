import sys

def nthugly(n):
    ugly = [1]
    i2 = i3 = i5= 0
    for ii in range(1,n):
        next_ugly = min(ugly[i2]*2, ugly[i3]*3, ugly[i5]*5 )
        ugly.append(next_ugly)

        if next_ugly == ugly[i2]*2:
            i2 += 1
        if next_ugly == ugly[i3]*3:
            i3 += 1
        if next_ugly == ugly[i5]*5:
            i5 += 1
    return ugly[-1]

for line in sys.stdin:
    a = line.split()
    out = nthugly(a)
    print(out)
