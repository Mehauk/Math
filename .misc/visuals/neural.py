from manim import *

class InputNodeOutput(Scene):
    def construct(self):
        # x = MathTex("x").scale(2).move_to(LEFT*5)
        # w = MathTex("w_0").scale(1.7).move_to(LEFT*4+UP*0.3)
        # line = Line(start=LEFT*5 + RIGHT/3, end=LEFT*3)
        # node = Circle(1).move_to(LEFT*2)
        # b = MathTex("b_0").scale(1.7).move_to(LEFT*2)
        # w1 = MathTex("w_1").scale(1.7).move_to(LEFT*0+UP*0.3)
        # line1 = Line(start=LEFT, end=RIGHT)
        # node1 = Circle(1).move_to(RIGHT*2)
        # b1 = MathTex("b_1").scale(1.7).move_to(RIGHT*2)
        # line2 = Arrow(start=RIGHT*3, end=RIGHT*5)
        # fx = MathTex("f(x)").scale(2).move_to(RIGHT*6)

        # node.color = TEAL

        # self.add(x, w, line, node, b, w1, line1, node1, b1, line2, fx)

        x = MathTex("x").scale(2).move_to(LEFT*3)
        a = MathTex("a").scale(1.7).move_to(LEFT*2+UP*0.3)
        w = MathTex("w_0").scale(1.7).move_to(LEFT*2+UP*0.3)
        w1 = MathTex("w_1").scale(1.7).move_to(LEFT*2+UP*0.3)
        line = Line(start=LEFT*3 + RIGHT/3, end=LEFT)
        node = Circle(1)
        b = MathTex("b_0").scale(1.7)
        b1 = MathTex("b_1").scale(1.7)
        line2 = Arrow(start=RIGHT, end=RIGHT*3)
        fx = MathTex("f(x)").scale(2).move_to(RIGHT*4)

        self.add(x, w, line, node, b, line2, fx)