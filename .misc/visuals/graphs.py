from manim import *
import math

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
        # graph = ImplicitFunction(
        #     lambda x, y: math.atan(x) - y,
        #     color=TEAL
        # )

        # plane = NumberPlane(background_line_style={"stroke_opacity": "0.1"})

        # plane.y_axis.scale(2)
        # plane.axes = Axes(           
        #     x_range=[0, 10, 1],
        #     y_range=[-2, 6, 1],
        #     tips=False,
        #     axis_config={"include_numbers": True},
        #     y_axis_config={"scaling": LogBase(custom_labels=True)},
        #     )
        # plane.y_axis.set_opacity(1)
        # plane.x_axis.set_opacity(0)

        # # graph.get_image(camera=Camera()).save("test.png")
        # self.add(plane, graph) 