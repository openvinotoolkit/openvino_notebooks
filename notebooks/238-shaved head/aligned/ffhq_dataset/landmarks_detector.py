import dlib


class LandmarksDetector:
    def __init__(self, predictor_model_path):
        """
        :param predictor_model_path: path to shape_predictor_68_face_landmarks.dat file
        """
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks
            # 导出onnx模型
            # input_names = ['input']
            # output_names = ['output']
            # # x = torch.randn (64, 3, 3, 3, requires_grad=True).cpu().numpy()
            # torch.onnx.export(self.shape_predictor,(img, detection), '../pd_model/shape_predictor_68_face_landmarks.onnx',
            #                    input_names=input_names, output_names=output_names,
            #                    dynamic_axes={"input": {0: "batch_size", 0: "h", 0: "w"}, "output": {0: "batch_size"}},
            #                    verbose='True', opset_version=11)
            # 导出onnx模型
