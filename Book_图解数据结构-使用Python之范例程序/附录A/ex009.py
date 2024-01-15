import sys

#声明字符串数组并初始化
newspaper=['1.北京晚报','2.作家文摘','3.参考消息', \
                          '4.证券报','5.不需要']
#字符串数组的输出
for i in range(5):
    print('%s  ' %newspaper[i], end='')

try :
    choice=int(input('请输入选择:'))
    #输入的判断
    if choice>=0 and choice<4:
        print('%s' %newspaper[choice-1])
        print('谢谢您的订购!!!')
    elif choice==5:
        print('感谢您的参与!!!')
    else:
        print('数字选项输入错误')

except ValueError:
    print('所输入的不是数字')