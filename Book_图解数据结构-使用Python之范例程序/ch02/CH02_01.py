Score=[87,66,90,65,70]
Total_Score=0
for count in range(5):
    print('第 %d 位学生的分数:%d' %(count+1,Score[count]))
    Total_Score+=Score[count]
print('-------------------------')
print('5位学生的总分:%d' %Total_Score)