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
        cam:   the position of camera.
    """
    
    def __init__(self, width=500, height=450, cam=[3, 2, -6]):
        # screen init
        self.WIDTH, self.HEIGHT = width, height
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 60
        self.screen = np.zeros((self.HEIGHT, self.WIDTH, 3), dtype=np.int8)
        self.Cam = cam
        self.create_objects()
        
        # set color
        self.Color = {'red' : (255, 0, 0), 
                      'green' : (0, 255, 0),
                      'blue' : (0, 0, 255)}        

    def draw(self):
        # Change the background color like this:
        # self.screen.fill(<Your defined color>))
        self.world_axes.draw()

        # draw grid
        self.grid.draw()
        
        # draw skeleton
        self.skeleton.draw()

    def create_objects(self):
        """
        In this function we create and initialize the coordinates and grid and the objects we need.
        So you can add your objects here, and draw them by adding code in draw() function.
        """
        # set the camera position.
        self.camera = Camera(self, self.Cam)
        self.projection = Projection(self)

    # add axes
    # this maybe unused if you do not want see the axes of object
        self.axes = Axes(self)
        self.axes.translate([0.7, 0.9, 0.7])
        self.world_axes = Axes(self)
        self.world_axes.Move_flag = False
        self.world_axes.scale(2.5)
        self.world_axes.translate([0.0001, 0.0001, 0.0001])

        # add grid
        self.grid = Grid(self, 1, 20)
        self.grid.Move_flag = False
        self.grid.scale(15)
        self.grid.translate([0.0001, 0.0001, 0.0001])
        
        # add skeleton
        self.skeleton = Skeleton(self)

    def image(self):
        self.screen.fill(0)
        self.draw()
        
        # return not encoded image data.
        return self.screen
                
        
    def run(self):
        while True:
            self.screen.fill(0)
            self.draw()

            cv2.imshow("image", self.screen)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
            



