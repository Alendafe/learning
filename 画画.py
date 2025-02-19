import turtle

# Setup the screen
screen = turtle.Screen()
screen.bgcolor("white")

# Create a turtle for the boy
boy = turtle.Turtle()
boy.shape("turtle")
boy.color("blue")
boy.speed(5)

# Create a turtle for the girl
girl = turtle.Turtle()
girl.shape("turtle")
girl.color("pink")
girl.speed(5)

# Draw the boy
def draw_boy():
    boy.penup()
    boy.goto(-150, 0)
    boy.pendown()

    # Draw the boy's head
    boy.begin_fill()
    boy.circle(50)  # Head
    boy.end_fill()

    # Draw the boy's eyes (transparent glasses)
    boy.penup()
    boy.goto(-170, 50)
    boy.pendown()
    boy.setheading(0)
    boy.circle(10)  # Left lens

    boy.penup()
    boy.goto(-130, 50)
    boy.pendown()
    boy.circle(10)  # Right lens

    boy.penup()
    boy.goto(-160, 50)
    boy.pendown()
    boy.goto(-140, 50)  # Draw the bridge of glasses

    # Draw the boy's smile
    boy.penup()
    boy.goto(-170, 30)
    boy.pendown()
    boy.setheading(-60)
    boy.circle(30, 120)  # Smile

# Draw the girl
def draw_girl():
    girl.penup()
    girl.goto(150, 0)
    girl.pendown()

    # Draw the girl's head
    girl.begin_fill()
    girl.circle(50)  # Head
    girl.end_fill()

    # Draw the girl's eyes (black glasses)
    girl.penup()
    girl.goto(130, 50)
    girl.pendown()
    girl.setheading(0)
    girl.circle(10)  # Left lens

    girl.penup()
    girl.goto(170, 50)
    girl.pendown()
    girl.circle(10)  # Right lens

    girl.penup()
    girl.goto(120, 50)
    girl.pendown()
    girl.goto(180, 50)  # Draw the bridge of glasses

    # Draw the girl's smile
    girl.penup()
    girl.goto(130, 30)
    girl.pendown()
    girl.setheading(-60)
    girl.circle(30, 120)  # Smile

# Call functions to draw the boy and the girl
draw_boy()
draw_girl()

# Hide turtles after drawing
boy.hideturtle()
girl.hideturtle()

# Finish drawing
turtle.done()
