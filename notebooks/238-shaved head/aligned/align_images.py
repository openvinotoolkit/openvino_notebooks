import os
import bz2
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
import matplotlib.pyplot as plt
LANDMARKS_MODEL_URL = 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'
def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path
if __name__ == "__main__":
    """
    Extracts and aligns all faces from images using DLib and a function from original FFHQ dataset preparation step
    python align_images.py /raw_images /aligned_images
    """
    # 调用人脸检测器
    #detector = dlib.get_frontal_face_detector ()

    # 加载 预测关键点模型（68个关键点）
    #predictor = dlib.shape_predictor ("shape_predictor_68_face_landmarks.dat")

    landmarks_model_path='./model/shape_predictor_68_face_landmarks.dat'
    # RAW_IMAGES_DIR = sys.argv[1]
    # ALIGNED_IMAGES_DIR = sys.argv[2]
    RAW_IMAGES_DIR ='../test_data/faceorigin/'
    ALIGNED_IMAGES_DIR =r'../test_data/origin/'

    landmarks_detector = LandmarksDetector(landmarks_model_path)
    for img_name in os.listdir(RAW_IMAGES_DIR):
        raw_img_path = os.path.join(RAW_IMAGES_DIR, img_name)
        for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(raw_img_path), start=1):
            face_img_name = '%s_%02d.png' % (os.path.splitext(img_name)[0], i)
            aligned_face_path = os.path.join(ALIGNED_IMAGES_DIR, face_img_name)
            print(face_landmarks.shape)
            plt.imshow (face_landmarks)
            plt.show ()
            image_align(raw_img_path, aligned_face_path, face_landmarks)

            # https: // github.com / github - luffy / PFLD_68points_Pytorch

            # model = torch.load ("D:/projects/PFLD_68points_Pytorch-master/pretrained_model/mobileNetV2_1.0.pth")
            # model.cpu ()
            # model.eval ()
            #
            # dummy_input1 = torch.randn (1, 3, 112, 112, dtype=torch.float)
            # torch.onnx.export (model, (dummy_input1), "mobilenetv2_pfld.onnx", verbose=True)

