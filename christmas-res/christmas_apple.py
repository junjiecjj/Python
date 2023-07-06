import turtle
import time

turtle.setup(800, 800)
turtle.speed(5)
turtle.pensize(5)

# 先画一个圆,并填充为红色
turtle.begin_fill()
turtle.color("red")
turtle.circle(radius=150)
turtle.end_fill()

# 画苹果把儿
turtle.color("brown")
turtle.pu()
turtle.goto(-90, 200)
turtle.pd()
turtle.circle(180, 40)
turtle.pu()
turtle.seth(105)
turtle.goto(-20, 220)
turtle.pd()
turtle.circle(180, 50)
turtle.pd()

# 画左边叶子
turtle.begin_fill()
turtle.color("green")
turtle.circle(180, 50)
turtle.seth(-30)
turtle.circle(180, 55)
turtle.end_fill()

# 画右边叶子
turtle.begin_fill()
turtle.color("green")
turtle.seth(0)
turtle.circle(180, 50)
turtle.seth(-180)
turtle.circle(180, 50)
turtle.end_fill()
turtle.done()
time.sleep(10)
