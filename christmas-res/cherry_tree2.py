import turtle as tt
from random import *
from math import *



def tree(n, l):
    tt.pd()  # 下笔
    # 阴影效果
    t = cos(radians(tt.heading() + 45)) / 8 + 0.25
    tt.pencolor(t, t, t)
    tt.pensize(n / 3)
    tt.forward(l)  # 画树枝

    if n > 0:
        b = random() * 15 + 10  # 右分支偏转角度
        c = random() * 15 + 10  # 左分支偏转角度
        d = l * (random() * 0.25 + 0.7)  # 下一个分支的长度
        # 右转一定角度,画右分支
        tt.right(b)
        tree(n - 1, d)
        # 左转一定角度，画左分支
        tt.left(b + c)
        tree(n - 1, d)
        # 转回来
        tt.right(c)
    else:
        # 画叶子
        tt.right(90)
        n = cos(radians(tt.heading() - 45)) / 4 + 0.5
        tt.pencolor(n, n * 0.8, n * 0.8)
        tt.circle(3)
        tt.left(90)
        # 添加0.3倍的飘落叶子
        if random() > 0.7:
            tt.pu()
            # 飘落
            t = tt.heading()
            an = -40 + random() * 40
            tt.setheading(an)
            dis = int(800 * random() * 0.5 + 400 * random() * 0.3 + 200 * random() * 0.2)
            tt.forward(dis)
            tt.setheading(t)
            # 画叶子
            tt.pd()
            tt.right(90)
            n = cos(radians(tt.heading() - 45)) / 4 + 0.5
            tt.pencolor(n * 0.5 + 0.5, 0.4 + n * 0.4, 0.4 + n * 0.4)
            tt.circle(2)
            tt.left(90)
            tt.pu()
            # 返回
            t = tt.heading()
            tt.setheading(an)
            tt.backward(dis)
            tt.setheading(t)
    tt.pu()
    tt.backward(l)  # 退回

def init():
    tt.bgcolor(0.5, 0.5, 0.5)  # 背景色
    tt.hideturtle()  # 隐藏turtle
    tt.speed(10)  # 速度 1-10渐进，0 最快
    tt.tracer(0, 0)
    tt.pu()  # 抬笔
    tt.backward(100)
    tt.left(90)  # 左转90度
    tt.pu()  # 抬笔
    tt.backward(300)  # 后退300
    tree(12, 100)  # 递归7层
    tt.done()

def main():
    init()

if __name__ == '__main__':
    main()