from manim import *

class GraphFunctionDots(Scene):
    def construct(self):
        dots = []
        lines = []
        for i in range(-3, 3):
            dots.append(Dot(point=i*i*UP + i*RIGHT + 2*DOWN))
            x = i + 1
            lines.append(Line(
                start=i*i*UP + i*RIGHT + 2*DOWN, 
                end=x*x*UP + x*RIGHT + 2*DOWN
                              ))


        plane = NumberPlane(
            background_line_style={"stroke_opacity": "0.1"}, 
            x_range=(-40, 40, 1),
            y_range=(-30, 30, 1),
            )


        curve = plane.plot(lambda x: x*x - 2)
        curve2 = plane.plot(lambda x: (x + 0.04)**2 - 2.04)
        curve.color = TEAL

        plane.move_to(DOWN*10 + LEFT*10)

        self.add(plane, curve, curve2, *dots)