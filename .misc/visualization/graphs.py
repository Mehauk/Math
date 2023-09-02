from manim import *

class GraphFunction(Scene):
    def construct(self):
        graph = ImplicitFunction(
            lambda x, y: x * x - y,
            color=YELLOW
        )
        self.add(NumberPlane(), graph)