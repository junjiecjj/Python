SIZE=7  
NUMBER=6
INFINITE=99999 # 无穷大  

Graph_Matrix=[[0]*SIZE for row in range(SIZE)] # 图的数组
distance=[0]*SIZE  # 路径长度数组

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
            

# 单点对全部顶点的最短距离  
def shortestPath(vertex1, vertex_total):
    shortest_vertex = 1   #记录最短距离的顶点
    goal=[0]*SIZE         #用来记录该顶点是否被选取
    for i in range(1,vertex_total+1):
        goal[i] = 0
        distance[i] = Graph_Matrix[vertex1][i]
    goal[vertex1] = 1
    distance[vertex1] = 0
    print()

    for i in range(1,vertex_total):
        shortest_distance = INFINITE
        for j in range(1,vertex_total+1):
            if goal[j]==0 and shortest_distance>distance[j]:
                shortest_distance=distance[j]
                shortest_vertex=j
            
        goal[shortest_vertex] = 1
        # 计算开始顶点到各顶点的最短距离 
        for j in range(vertex_total+1):
            if goal[j] == 0 and \
               distance[shortest_vertex]+Graph_Matrix[shortest_vertex][j] \
               <distance[j]:
                distance[j]=distance[shortest_vertex] \
                +Graph_Matrix[shortest_vertex][j]

# 主程序
global Path_Cost
Path_Cost = [ [1, 2, 29], [2, 3, 30],[2, 4, 35], \
              [3, 5, 28],[3, 6, 87],[4, 5, 42], \
              [4, 6, 75],[5, 6, 97]]

BuildGraph_Matrix(Path_Cost)
shortestPath(1,NUMBER) # 搜索最短路径 
print('-----------------------------------')
print('顶点1到各顶点最短距离的最终结果')
print('-----------------------------------')
for j in range(1,SIZE):
    print('顶点 1到顶点%2d的最短距离=%3d' %(j,distance[j]))
print('-----------------------------------')
print()