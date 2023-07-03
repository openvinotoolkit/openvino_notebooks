from analog.base import analog_base
import numpy as np
import cv2
import os


class analog_paddle(analog_base):
    def __init__(self, config, output_dir):
        super().__init__(config, output_dir)

    def detect(self, input):
        # Prepare the input data for meter detection model
        im_shape = np.array(
            [[self.input_shape, self.input_shape]]).astype('float32')
        scale_factor = np.array([[1, 2]]).astype('float32')
        input_image = self.det_preprocess(input, self.input_shape)
        inputs_dict = {'image': input_image,
                       "im_shape": im_shape, "scale_factor": scale_factor}

        # Run meter detection model
        det_results = self.det_compiled_model(
            inputs_dict)[self.det_output_layer]

        # Filter out the bounding box with low confidence
        filtered_results = self.filter_bboxes(
            det_results, self.score_threshold)

        # Prepare the input data for meter segmentation model
        scale_x = input.shape[1] / self.input_shape * 2
        scale_y = input.shape[0] / self.input_shape

        # Create the individual picture for each detected meter
        roi_imgs, self.loc = self.roi_crop(
            input, filtered_results, scale_x, scale_y)
        roi_imgs, resize_imgs = self.roi_process(roi_imgs, self.METER_SHAPE)

        # Create the pictures of detection results
        roi_stack = np.hstack(resize_imgs)

        cv2.imwrite(os.path.join(self.output_dir, "detection_results.jpg"), roi_stack)

        return roi_imgs

    def segment(self, input):
        seg_results = list()
        num_imgs = len(input)
        image_list = list()

        # Run meter segmentation model on all detected meters
        for i in range(0, num_imgs, self.seg_batch_size):
            batch = input[i: min(num_imgs, i + self.seg_batch_size)]
            seg_result = self.seg_compiled_model({"image": np.array(batch)})[
                self.seg_output_layer]
            seg_results.extend(seg_result)
        results = []
        for i in range(len(seg_results)):
            results.append(np.argmax(seg_results[i], axis=0))
        seg_results = self.erode(results, self.erode_kernel)

        for i in range(len(seg_results)):
            image_list.append(self.segmentation_map_to_image(
                seg_results[i], self.COLORMAP))

        # Create the pictures of segmentation results
        mask_stack = np.hstack(image_list)

        cv2.imwrite(os.path.join(self.output_dir, "segmentation_results.jpg"),
                    cv2.cvtColor(mask_stack, cv2.COLOR_RGB2BGR))

        return seg_results

    def filter_bboxes(self, det_results, score_threshold):
        """
        Filter out the detection results with low confidence

        Param：
            det_results (list[dict]): detection results
            score_threshold (float)： confidence threshold

        Retuns：
            filtered_results (list[dict]): filter detection results
        
        """
        filtered_results = []
        for i in range(len(det_results)):
            if det_results[i, 1] > score_threshold:
                filtered_results.append(det_results[i])
        return filtered_results

    def roi_crop(self, image, results, scale_x, scale_y):
        """
        Crop the area of detected meter of original image

        Param：
            img (np.array)：original image。
            det_results (list[dict]): detection results
            scale_x (float): the scale value in x axis
            scale_y (float): the scale value in y axis

        Retuns：
            roi_imgs (list[np.array]): the list of meter images
            loc (list[int]): the list of meter locations
        
        """
        roi_imgs = []
        loc = []
        for result in results:
            bbox = result[2:]
            xmin, ymin, xmax, ymax = [int(
                bbox[0] * scale_x), int(bbox[1] * scale_y), int(bbox[2] * scale_x), int(bbox[3] * scale_y)]
            sub_img = image[ymin:(ymax + 1), xmin:(xmax + 1), :]
            roi_imgs.append(sub_img)
            loc.append([xmin, ymin, xmax, ymax])
        return roi_imgs, loc
