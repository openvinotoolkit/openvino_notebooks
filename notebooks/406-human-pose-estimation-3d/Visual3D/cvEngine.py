# This is 3d Engine
from object_3d import *
from camera import *
from projection import *
from IPython.display import display, Image, clear_output
import cv2

class Engine3D:
    """
    Summary of class here.

    This is an OpencV graphics interface to complete the graphics rendering pipeline.
    
    Attributes:
        width, height:   Size of the view window.
    """
    
    def __init__(self, width=500, height=450):
        # screen init
        self.WIDTH, self.HEIGHT = width, height
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 60
        self.screen = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.int8)
        self.create_objects()
        
        # set color
        self.Color = {'red' : (255, 0, 0), 
                      'green' : (0, 255, 0),
                      'blue' : (0, 0, 255)}
        
        # object
        # self.world_axes = world_axes
        
        # pop_up controls
        # self.key_callback = Camera.control();
        

    def draw(self):
        # self.screen.fill(pg.Color('lightgoldenrod4'))
        self.world_axes.draw()
        # self.axes.draw()
        # self.tetrahedron.draw()

        # draw grid
        self.grid.draw()
        
        # draw skeleton
        self.skeleton.draw()

    def create_objects(self):
        """
        In this function we create and initialize the coordinates and grid and the objects we need.
        """
        self.camera = Camera(self, [2, 3, -6])
        # self.camera = Camera(self, [5, 2, -3])
        self.projection = Projection(self)
        # self.tetrahedron = Object3D(self)
        # self.tetrahedron.translate([0.2, 0.4, 0.2])
        # self.object.rotate_y(math.pi / 6)

        self.axes = Axes(self)
        self.axes.translate([0.7, 0.9, 0.7])
        self.world_axes = Axes(self)
        self.world_axes.Move_flag = False
        self.world_axes.scale(2.5)
        self.world_axes.translate([0.0001, 0.0001, 0.0001])

        # add grid
        self.grid = Grid(self, 1, 20)
        self.grid.Move_flag = False
        self.grid.scale(10)
        self.grid.translate([0.0001, 0.0001, 0.0001])
        
        # add skeleton
        self.skeleton = Skeleton(self)

    def image(self):
        self.screen.fill(0)
        self.draw()
        # _, encoded_img = cv2.imencode(".jpg", self.screen, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
        # return Image(data=encoded_img)
        return self.screen
        # display(Image(data=encoded_img))
                
        
    def run(self):
        while True:
            self.screen.fill(0)
            self.draw()

            cv2.imshow("image", self.screen)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
            



# if __name__ == '__main__':
#     newEngine = engine3D()
#     # newEngine.run()
#     newEngine.image()
