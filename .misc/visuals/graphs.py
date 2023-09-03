from manim import *

class GraphFunctionDots(Scene):
    def construct(self):
        plane = NumberPlane(
            background_line_style={"stroke_opacity": "0.1"}, 
            x_range=(0, 40, 1),
            y_range=(0, 30, 1),
            )

        dots = []
        for i in range(-3, 5):
            c = TEAL
            if i < 0 or i > 3: c = GRAY
            dots.append(Dot(point=i*(UP + RIGHT) + DOWN, color=c))

        plane.move_to(DOWN*10 + LEFT*10)

        self.add(plane, *dots)

class GraphFunction(Scene):
    def construct(self):
        plane = NumberPlane(
            background_line_style={"stroke_opacity": "0.1"}, 
            x_range=(-40, 40, 1),
            y_range=(-30, 30, 1),
            )

        curve = plane.plot(lambda x: x*x)
        curve.color = TEAL

        plane.move_to(DOWN*10 + LEFT*10)

        self.add(plane, curve)