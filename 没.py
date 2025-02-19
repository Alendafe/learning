import turtle
import random
import time

# 初始化窗口
window = turtle.Screen()
window.bgcolor("black")
window.title("Python Fireworks Pro")
window.setup(width=800, height=600)
window.tracer(0, 0)  # 禁用自动刷新

# 获取窗口实际边界
WIN_WIDTH = window.window_width() // 2
WIN_HEIGHT = window.window_height() // 2


class Firework:
    def __init__(self, x):
        self.particles = []
        self.exploded = False
        self.launch(x)

    def launch(self, x):
        # 初始化火箭（从底部发射）
        self.rocket = turtle.Turtle(shape="circle")
        self.rocket.shapesize(0.5)
        self.rocket.color("white")
        self.rocket.penup()
        self.rocket.goto(x, -WIN_HEIGHT + 50)  # 从底部上方50像素发射

        # 动态计算爆炸高度（窗口上半部随机位置）
        self.target_y = random.randint(100, WIN_HEIGHT - 100)
        self._animate_ascend()

    def _animate_ascend(self):
        try:
            current_y = self.rocket.ycor()
            if current_y < self.target_y:
                self.rocket.sety(current_y + 8)
                window.ontimer(self._animate_ascend, 30)
            else:
                self.explode()
                self.rocket.hideturtle()
        except turtle.Terminator:
            return

    def explode(self):
        colors = ["red", "yellow", "cyan", "pink", "orange"]
        explosion_x = self.rocket.xcor()
        explosion_y = self.rocket.ycor()

        # 生成爆炸粒子（限制在窗口范围内）
        for _ in range(80):
            angle = random.uniform(0, 360)
            speed = random.uniform(3, 6)  # 降低速度防止出界
            dx = speed * turtle.Vec2D(1, 0).rotate(angle)[0]
            dy = speed * turtle.Vec2D(1, 0).rotate(angle)[1]

            p = ExplosionParticle(
                color=random.choice(colors),
                start_x=explosion_x,
                start_y=explosion_y,
                dx=dx,
                dy=dy
            )
            self.particles.append(p)
        self.exploded = True

    def update(self):
        for p in self.particles[:]:
            try:
                p.update()
                if not p.alive:
                    self.particles.remove(p)
            except:
                pass


class ExplosionParticle(turtle.Turtle):
    def __init__(self, color, start_x, start_y, dx, dy):
        super().__init__(shape="circle")
        self.shapesize(0.3)
        self.color(color)
        self.penup()
        self.goto(start_x, start_y)
        self.dx = dx
        self.dy = dy
        self.lifespan = random.randint(80, 120)
        self.alive = True
        self.tail = []
        self.max_tail_length = 5

    def update(self):
        if self.alive:
            # 添加边界检测
            if abs(self.xcor()) > WIN_WIDTH or abs(self.ycor()) > WIN_HEIGHT:
                self.alive = False
                return

            # 物理模拟
            self.dx *= 0.97
            self.dy -= 0.3

            new_x = self.xcor() + self.dx
            new_y = self.ycor() + self.dy

            # 记录轨迹点
            self.tail.append((new_x, new_y))
            if len(self.tail) > self.max_tail_length:
                self.tail.pop(0)

            # 绘制轨迹
            self._draw_trail()
            self.goto(new_x, new_y)

            self.lifespan -= 2
            if self.lifespan <= 0:
                self.alive = False
                self._clear_trail()
                self.hideturtle()

    def _draw_trail(self):
        for i, (x, y) in enumerate(self.tail):
            if i > 0:
                prev_x, prev_y = self.tail[i - 1]
                self._draw_segment(prev_x, prev_y, x, y, alpha=i / self.max_tail_length)

    def _draw_segment(self, x1, y1, x2, y2, alpha):
        try:
            t = turtle.Turtle(visible=False)
            t.penup()
            t.goto(x1, y1)
            t.color(self.color()[0], (self.color()[1][0], self.color()[1][1], self.color()[1][2], alpha))
            t.pensize(2)
            t.pendown()
            t.goto(x2, y2)
            t._turtle._canvas.after(50, t.reset)
        except:
            pass


# 全局管理
active_fireworks = []


def launch_firework(x, y):
    if len(active_fireworks) < 5:
        fw = Firework(x)  # 只使用x坐标
        active_fireworks.append(fw)


window.onclick(launch_firework)

try:
    while True:
        # 更新所有烟花
        for fw in active_fireworks[:]:
            fw.update()
            if fw.exploded and not fw.particles:
                active_fireworks.remove(fw)

        window.update()
        time.sleep(0.02)

except Exception as e:
    print("程序安全退出:", e)
    turtle.bye()