# 下三角矩阵 
global ARRAY_SIZE #矩阵的维数大小
ARRAY_SIZE=5
#一维数组的数组声明
num=int(ARRAY_SIZE*(1+ARRAY_SIZE)/2)
B=[None]*(num+1)

def getValue(i,j):
    index = int(ARRAY_SIZE*i-i*(i+1)/2+j)
    return B[index]

#下三角矩阵的内容
A=[[76, 0, 0, 0, 0], 
   [54, 51, 0, 0, 0], 
   [23, 8, 26, 0, 0], 
   [43, 35, 28, 18, 0], 
   [12, 9, 14, 35, 46]] 
  
print("==========================================\n")
print("下三角形矩阵：")
for i in range(ARRAY_SIZE):
    for j in range(ARRAY_SIZE):
        print('%d' %A[i][j],end='\t')
    print()
		
#将左下三角矩阵压缩为一维数组 */
index=0
for i in range(ARRAY_SIZE):
    for j in range(ARRAY_SIZE):
        if(A[i][j]!=0):
            index=index+1
            B[index]=A[i][j]
print("==========================================\n")
print("以一维的方式表示：\n")
print('[',end='')
for i in range(ARRAY_SIZE):
    for j in range(i+1,ARRAY_SIZE+1):
        print(' %d' %getValue(i,j),end='')
print(' ]')