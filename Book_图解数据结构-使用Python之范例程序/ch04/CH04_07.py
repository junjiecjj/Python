def hanoi(n, p1, p2, p3):
    if n==1: # 递归出口
        print('盘子从 %d 移到 %d' %(p1, p3))
    else:
        hanoi(n-1, p1, p3, p2)
        print('盘子从 %d 移到 %d' %(p1, p3))
        hanoi(n-1, p2, p1, p3)

j=int(input('请输入要移动盘子的数量：'))
hanoi(j,1, 2, 3)