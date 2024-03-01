import turtle
import time

screen = turtle.Screen()
screen.setup(700, 700)
circle = turtle.Turtle()
circle.shape('circle')
circle.color('red')
circle.speed('fastest')
circle.up()
square = turtle.Turtle()
square.shape('square')
square.color('green')
square.speed('fastest')
square.up()
circle.goto(0, 280)
circle.stamp()
k = 0
for i in range(1, 13):
    y = 30 * i
    for j in range(i - k):
        x = 30 * j
        square.goto(x, -y + 280)
        square.stamp()
        square.goto(-x, -y + 280)
        square.stamp()
if i % 4 == 0:
    x = 30 * (j + 1)
    circle.color('red')
    circle.goto(-x, -y + 280)
    circle.stamp()
    circle.goto(x, -y + 280)
    circle.stamp()
    k += 3
if i % 4 == 3:
    x = 30 * (j + 1)
    circle.color('yellow')
    circle.goto(-x, -y + 280)
    circle.stamp()
    circle.goto(x, -y + 280)
    circle.stamp()
square.color('brown')
for i in range(13, 17):
    y = 30 * i
    for j in range(2):
        x = 30 * j
        square.goto(x, -y + 280)
        square.stamp()
        square.goto(-x, -y + 280)
        square.stamp()

time.sleep(5)