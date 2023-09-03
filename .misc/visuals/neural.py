from manim import *

class InputNodeOutput(Scene):
    def construct(self):
        x = MathTex("x").scale(2).move_to(LEFT*3)
        line = Line(start=LEFT*3 + RIGHT/3, end=LEFT)
        node = Circle(1)
        n = MathTex("n").scale(1.7)
        line2 = Line(start=RIGHT, end=RIGHT*3)
        fx = MathTex("f(x)").scale(2).move_to(RIGHT*4)

        node.color = TEAL

        self.add(x, line, node, n, line2, fx)