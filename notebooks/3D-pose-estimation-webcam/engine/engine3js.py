from IPython.display import display
from pythreejs import *


class Engine3js:
    """
    Summary of class here.

    The implementation of these interfaces depends on Pythreejs, so make
    sure you install and authorize the Jupyter Widgets plug-in correctly.

    Attributes:
        view_width, view_height:   Size of the view window.
        position:                  Position of the camera.
        lookAtPos:                 The point at which the camera looks at.
        axis_size:                 The size of axis.
        grid_length:               grid size, length == width.
        grid_num:                  grid number.
        grid:                      If use grid.
        axis:                      If use axis.
    """

    def __init__(
        self,
        view_width=455,
        view_height=256,
        position=[300, 100, 0],
        lookAtPos=[0, 0, 0],
        axis_size=300,
        grid_length=600,
        grid_num=20,
        grid=False,
        axis=False,
    ):
        self.view_width = view_width
        self.view_height = view_height
        self.position = position

        # set the camera
        self.cam = PerspectiveCamera(position=self.position, aspect=self.view_width / self.view_height)
        self.cam.lookAt(lookAtPos)

        # x,y,z axis
        self.axis = AxesHelper(axis_size)  # axes length

        # set grid size
        self.gridHelper = GridHelper(grid_length, grid_num)

        # set scene
        self.scene = Scene(
            children=[
                self.cam,
                DirectionalLight(position=[3, 5, 1], intensity=0.6),
                AmbientLight(intensity=0.5),
            ]
        )

        # add axis or grid
        if grid:
            self.scene.add(self.gridHelper)

        if axis:
            self.scene.add(self.axis)

        # render the objects in scene
        self.renderer = Renderer(
            camera=self.cam,
            scene=self.scene,
            controls=[OrbitControls(controlling=self.cam)],
            width=self.view_width,
            height=self.view_height,
        )
        # display(renderer4)

    def get_width(self):
        return self.view_width

    def plot(self):
        self.renderer.render(self.scene, self.cam)

    def scene_add(self, object):
        self.scene.add(object)

    def scene_remove(self, object):
        self.scene.remove(object)


class Geometry:
    """
    This is the geometry base class that defines buffer and material.
    """

    def __init__(self, name="geometry"):
        self.geometry = None
        self.material = None
        self.name = name

    def get_Name():
        return self.name


class Skeleton(Geometry):
    """
    This is the class for drawing human body poses.
    """

    def __init__(self, name="skeleton", lineWidth=3, body_edges=[]):
        super(Skeleton, self).__init__(name)
        self.material = LineBasicMaterial(vertexColors="VertexColors", linewidth=lineWidth)
        self.colorSet = BufferAttribute(
            np.array(
                [
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [1, 0, 0],
                    [1, 0, 0],
                    [0, 1, 0],
                ],
                dtype=np.float32,
            ),
            normalized=False,
        )
        # self.geometry.attributes["color"] = self.colorSet
        self.body_edges = body_edges

    def __call__(self, poses_3d):
        poses = []
        for pose_position_tmp in poses_3d:
            bones = []
            for edge in self.body_edges:
                # put pair of points as limbs
                bones.append(pose_position_tmp[edge[0]])
                bones.append(pose_position_tmp[edge[1]])

            bones = np.asarray(bones, dtype=np.float32)

            # You can find the api in https://github.com/jupyter-widgets/pythreejs

            self.geometry = BufferGeometry(
                attributes={
                    "position": BufferAttribute(bones, normalized=False),
                    # It defines limbs' color
                    "color": self.colorSet,
                }
            )

            pose = LineSegments(self.geometry, self.material)
            poses.append(pose)
            # self.geometry.close()
        return poses

    def plot(self, pose_points=None):
        return self.__call__(pose_points)


class Cloudpoint(Geometry):
    """
    This is the class for drawing cloud points.
    """

    def __init__(self, name="cloudpoint", points=[], point_size=5, line=None, points_color="blue"):
        super(Cloudpoint, self).__init__(name)
        self.material = PointsMaterial(size=point_size, color=points_color)
        self.points = points
        self.line = line

    def __call__(self, points_3d):
        self.geometry = BufferGeometry(
            attributes={
                "position": BufferAttribute(points_3d, normalized=False),
                # It defines points' vertices' color
                "color": BufferAttribute(
                    np.array(
                        [
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                        ],
                        dtype=np.float32,
                    ),
                    normalized=False,
                ),
            },
        )
        cloud_points = Points(self.geometry, self.material)

        if self.line is not None:
            g1 = BufferGeometry(
                attributes={
                    "position": BufferAttribute(line, normalized=False),
                    # It defines limbs' color
                    "color": BufferAttribute(
                        # Here you can set vertex colors, if you set the 'color' option = vertexes
                        np.array(
                            [
                                [1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                                [0, 0, 1],
                                [1, 0, 0],
                                [1, 0, 0],
                                [0, 1, 0],
                            ],
                            dtype=np.float32,
                        ),
                        normalized=False,
                    ),
                },
            )
            m1 = LineBasicMaterial(color="red", linewidth=3)
            facemesh = LineSegments(g1, m1)
            return [cloud_points, facemesh]

        return cloud_points


def Box_bounding(Geometry):
    def __init__(self, name="Box", lineWidth=3):
        super(Box_bounding, self).__init__(name)
        self.material = LineBasicMaterial(vertexColors="VertexColors", linewidth=lineWidth)
        self.edge = []

    def __call__(self, points=None):
        pass
