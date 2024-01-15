arr=[[0]*6 for row in range(6)]  #声明矩阵arr
    
data=[[1,2],[2,1],[2,3],[2,4],[4,3],[4,1]]  #图各边的起点值和终点值
for i in range(6):      #读取图的数据
    for j in range(6):  #填入arr矩阵
        tmpi=data[i][0]     #tmpi为起始顶点
        tmpj=data[i][1]     #tmpj为终止顶点
        arr[tmpi][tmpj]=1   #有边的点填入1

print('有向图矩阵：')
for i in range(1,6):
    for j in range(1,6):
        print('[%d] ' %arr[i][j],end='')  #打印矩阵内容
    print()