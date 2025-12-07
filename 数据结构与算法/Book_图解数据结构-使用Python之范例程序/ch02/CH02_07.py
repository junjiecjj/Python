NONZERO=0
temp=1
Sparse=[[15,0,0,22,0,-15],[0,11,3,0,0,0],
        [0,0,0,-6,0,0],[0,0,0,0,0,0],
        [91,0,0,0,0,0],[0,0,28,0,0,0]] #声明稀疏矩阵,稀疏矩阵的所有元素设为0
Compress=[[None] * 3 for row in range(9)] #声明压缩矩阵
	
print('[稀疏矩阵的各个元素]') #打印出稀疏矩阵的各个元素
for i in range(6):
    for j in range(6):
        print('[%d]' %Sparse[i][j], end='\t')
        if Sparse[i][j] !=0:
            NONZERO=NONZERO+1
    print()

#开始压缩稀疏矩阵
Compress[0][0] = 6
Compress[0][1] = 6
Compress[0][2] = NONZERO

for i in range(6):
    for j in range(6):
        if Sparse[i][j] !=0:
            Compress[temp][0]=i
            Compress[temp][1]=j
            Compress[temp][2]=Sparse[i][j]
            temp=temp+1
	
print('[稀疏矩阵压缩后的内容]') #打印出压缩矩阵的各个元素
for i in range(NONZERO+1):
    for j in range(3):
        print('[%d] ' %Compress[i][j], end='')
    print()