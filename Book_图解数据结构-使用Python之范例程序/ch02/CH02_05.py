#[示范]:运算两个矩阵相乘的结果

def MatrixMultiply(arrA, arrB,arrC,M,N,P):
    global C
    if M<=0 or N<=0 or P<=0:
        print('[错误:维数M,N,P必须大于0]')
        return
    for i in range(M):
        for j in range(P):
            Temp=0
            for k in range(N):
                Temp = Temp + int(arrA[i*N+k])*int(arrB[k*P+j])
            arrC[i*P+j] = Temp

print('请输入矩阵A的维数(M,N): ')
M=int(input('M= '))
N=int(input('N= '))
A=[None]*M*N #声明大小为MxN的列表A

print('[请输入矩阵A的各个元素]')
for i in range(M):
    for j in range(N):
        A[i*N+j]=input('a%d%d='%(i,j))

print('请输入矩阵B的维数(N,P): ')
N=int(input('N= '))
P=int(input('P= '))

B=[None]*N*P #声明大小为NxP的列表B

print('[请输入矩阵B的各个元素]')
for i in range(N):
    for j in range(P):
        B[i*P+j]=input('b%d%d='%(i,j))

C=[None]*M*P #声明大小为MxP的列表C
MatrixMultiply(A,B,C,M,N,P)
print('[AxB的结果是]')
for i in range(M):
    for j in range(P):
        print('%d' %C[i*P+j], end='\t')
    print()