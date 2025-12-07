SIZE=7  
NUMBER=6
INFINITE=99999 # 无穷大  

Graph_Matrix=[[0]*SIZE for row in range(SIZE)] # 图的数组
distance=[[0]*SIZE for row in range(SIZE)] # 路径长度数组

# 建立图
def BuildGraph_Matrix(Path_Cost):
    for i in range(1,SIZE):
        for j in range(1,SIZE):
            if i == j :
                Graph_Matrix[i][j] = 0 # 对角线设为0
            else:
                Graph_Matrix[i][j] = INFINITE
    # 存入图的边
    i=0
    while i<SIZE:
        Start_Point = Path_Cost[i][0]
        End_Point = Path_Cost[i][1]
        Graph_Matrix[Start_Point][End_Point]=Path_Cost[i][2]
        i+=1
        
# 打印出图 
 
def shortestPath(vertex_total):
    # 初始化图的长度数组
    for i in range(1,vertex_total+1):
        for j in range(i,vertex_total+1):
            distance[i][j]=Graph_Matrix[i][j]
            distance[j][i]=Graph_Matrix[i][j]

    # 使用Floyd算法找出所有顶点两两之间的最短距离
    for k in range(1,vertex_total+1):
        for i in range(1,vertex_total+1):
            for j in range(1,vertex_total+1):
                if distance[i][k]+distance[k][j]<distance[i][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
            

Path_Cost = [[1, 2,20],[2, 3, 30],[2, 4, 25], \
             [3, 5, 28],[4, 5, 32],[4, 6, 95],[5, 6, 67]] 
BuildGraph_Matrix(Path_Cost)
print('===============================================')
print('      所有顶点两两之间的最短距离: ')
print('===============================================')
shortestPath(NUMBER) # 计算所有顶点间的最短路径 
#求得两两顶点间的最短路径长度数组后，将其打印出来
print('      顶点1  顶点2  顶点3  顶点4  顶点5  顶点6')
for i in range(1,NUMBER+1):
    print('顶点%d' %i, end='')
    for j in range(1,NUMBER+1):
        print('%6d ' %distance[i][j],end='')
    print()
print('===============================================')
print()