month=int(input('请输入月份: '))    
if 2<=month and month<=4:
    print('充满生机的春天')    
elif 5<=month and month<=7:
    print('热力四射的夏季')
elif month>=8 and month <=10:
    print('落叶缤纷的秋季')
elif month==1 or (month>=11 and month<=12):
    print('寒风刺骨的冬季')
else:
    print('很抱歉没有这个月份!!!')