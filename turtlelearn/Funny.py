from turtle import*
setup(600,600,200,200)
#脸
penup()
goto(-210,0)
seth(-90)
pendown()
pencolor('orange')
pensize(4)
begin_fill()
circle(210,360)
fillcolor('yellow')
end_fill()
pencolor('black')
#画嘴巴
pensize(5)
penup()
goto(-150,-30)
pendown()
seth(-90)
circle(150,180)
#左眼眶
penup()
pensize(4)
goto(-180,90)
pendown()
seth(40)
begin_fill()
circle(-120,80)
penup()
goto(-180,90)
seth(-130)
pendown()
circle(15,110)
seth(40)
circle(-106,83)
seth(30)
circle(18,105)
fillcolor('white')
end_fill()
#右眼眶
penup()
goto(20,90)
pendown()
seth(40)
begin_fill()
circle(-120,80)
penup()
goto(20,90)
pendown()
seth(-130)
circle(15,110)
seth(40)
circle(-106,83)
seth(30)
circle(18,105)
fillcolor('white')
end_fill()
#画眼珠
pensize(2)
penup()
goto(50,95)
pendown()
begin_fill()
circle(8,360)
fillcolor('black')
end_fill()
penup()
goto(-150,95)
pendown()
begin_fill()
circle(8,360)
fillcolor('black')
end_fill()
#画腮红
pensize(1)
pencolor('pink')
begin_fill()
penup()
goto(-160,50)
pendown()
seth(-90)
for i in range(2):
    for j in range(10):
        forward(j)
        left(9)
    for j in range(10,0,-1):
        forward(j)
        left(9)
fillcolor('pink')
end_fill()
pensize(1)
pencolor('pink')
begin_fill()
penup()
goto(40,50)
pendown()
seth(-90)
for i in range(2):
    for j in range(10):
        forward(j)
        left(9)
    for j in range(10,0,-1):
        forward(j)
        left(9)
fillcolor('pink')
end_fill()
hideturtle()
