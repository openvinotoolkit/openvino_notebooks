"""
As a standalone visualization library, you can create Windows to display 3D objects by calling the interface as shown in the following code.
"""
# This is 3d Engine
from object_3d import *
from camera import *
from projection import *
import cv2

class engine3D:
    def __init__(self):
        # pg.init()
        self.RES = self.WIDTH, self.HEIGHT = 600, 900
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 60
        self.screen = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.int8)
        # self.screen = pg.display.set_mode(self.RES)
        # self.clock = pg.time.Clock()
        self.create_objects()

    def draw(self):
        # self.screen.fill(pg.Color('lightgoldenrod4'))
        self.world_axes.draw()
        self.axes.draw()
        self.object.draw()

        # draw grid
        self.grid.draw()

    def create_objects(self):
        self.camera = Camera(self, [0.5, 1, -4])
        # self.camera = Camera(self, [5, 2, -3])
        self.projection = Projection(self)
        self.object = Object3D(self)
        self.object.translate([0.2, 0.4, 0.2])
        # self.object.rotate_y(math.pi / 6)

        self.axes = Axes(self)
        self.axes.translate([0.7, 0.9, 0.7])
        self.world_axes = Axes(self)
        self.world_axes.Move_flag = False
        self.world_axes.scale(2.5)
        self.world_axes.translate([0.0001, 0.0001, 0.0001])

        # add grid
        self.grid = Grid(self, 1, 10)
        self.grid.Move_flag = False
        self.grid.scale(5)
        self.grid.translate([0.0001, 0.0001, 0.0001])

    def run(self):
        while True:

            self.draw()
            # self.camera.control()
            # [exit() for i in pg.event.get() if i.type == pg.QUIT]
            # pg.display.set_caption(str(self.clock.get_fps()))
            # pg.display.flip()
            # self.clock.tick(self.FPS)

            cv2.imshow("image", self.screen)
            key = cv2.waitKey(20)
            if key == ord('c'):
                break
            self.screen.fill(0)



if __name__ == '__main__':
    newEngine = engine3D()
    newEngine.run()
