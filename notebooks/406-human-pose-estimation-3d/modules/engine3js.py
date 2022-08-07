from IPython.display import display
from pythreejs import *

class Engine3js:
    # view_width=600, view_height=400
    def __init__(self, view_width=455, view_height=256, position=[300, 100, 0], lookAtPos=[0, 0, 0], axis_size=300):
        
        self.view_width = view_width
        self.view_height = view_height
        self.position = position
        
        # set the camera
        self.cam = PerspectiveCamera(position=self.position, aspect=self.view_width / self.view_height)
        self.cam.lookAt(lookAtPos)

        # x,y,z axis
        self.axis = AxesHelper(axis_size)  # axes length

        # set grid size
        self.gridHelper = GridHelper(600, 20)

        # set scene
        self.scene = Scene(
            children=[
                self.cam,
                self.gridHelper,
                self.axis,
                DirectionalLight(position=[3, 5, 1], intensity=0.6),
                AmbientLight(intensity=0.5),
            ]
        )

        # scene = draw_skeleton(scene, tmp_poses_3d)
        # render the objects in scene
        
        self.renderer = Renderer(
            camera=self.cam,
            scene=self.scene,
            controls=[OrbitControls(controlling=self.cam)],
            width=self.view_width,
            height=self.view_height,
        )
        # display(renderer4)
        
    def get_width():
        return self.view_width

# import gc
def draw_skeleton(tmp_poses_3d, body_edges):
    # pose_position = tmp_poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]

    lines = []
    for pose_position_tmp in tmp_poses_3d:
        bones = []
        for edge in body_edges:
            bones.append(pose_position_tmp[edge[0]])
            bones.append(pose_position_tmp[edge[1]])

        bones = np.asarray(bones, dtype=np.float32)
        
        # You can find the api in https://github.com/jupyter-widgets/pythreejs
        g1 = BufferGeometry(
            attributes={
                "position": BufferAttribute(bones, normalized=False),
                # It defines limbs' color
                "color": BufferAttribute(np.array([
                        [1, 0, 0], [1, 0, 0],[0, 1, 0], [0, 0, 1],[1, 0, 0], [1, 0, 0],[0, 1, 0], [0, 0, 1],
                        [1, 0, 0], [1, 0, 0],[0, 1, 0], [0, 0, 1],[1, 0, 0], [1, 0, 0],[0, 1, 0], [0, 0, 1],
                        [1, 0, 0], [1, 0, 0],[0, 1, 0]
                    ], dtype=np.float32), normalized=False),
            },
        )
        m1 = LineBasicMaterial(vertexColors='VertexColors', linewidth=10)
        # m1 = LineBasicMaterial(color="red", linewidth=10)
        line = LineSegments(g1, m1)
        lines.append(line)
        # scene.add(line)
#     del g1, m1, line, bones
#     gc.collect()

    return lines