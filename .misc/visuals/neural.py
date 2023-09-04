from manim import *

class InputNodeOutput(Scene):
    def construct(self):
        x = MathTex("x").scale(2).move_to(LEFT*3)
        a = MathTex("a").scale(1.7).move_to(LEFT*2+UP*0.3)
        w = MathTex("w").scale(1.7).move_to(LEFT*2+UP*0.3)
        numa = MathTex("2.7").scale(1.5).move_to(LEFT*2+UP*0.3)
        line = Line(start=LEFT*3 + RIGHT/3, end=LEFT)
        node = Circle(1)
        b = MathTex("b").scale(1.7)
        n = MathTex("n").scale(1.7)
        numb = MathTex("1.8").scale(1.5)
        line2 = Arrow(start=RIGHT, end=RIGHT*3)
        fx = MathTex("f(x)").scale(2).move_to(RIGHT*4)

        node.color = TEAL

        self.add(x, w, line, node, b, line2, fx)