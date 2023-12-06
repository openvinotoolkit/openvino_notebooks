import cv2
import os
import numpy as np
import math
import openvino.runtime as ov


class analog_base(object):
    def __init__(self, config, output_dir):
        self.METER_SHAPE = [512, 512]
        self.CIRCLE_CENTER = [256, 256]
        self.CIRCLE_RADIUS = 250
        self.PI = math.pi
        self.RECTANGLE_HEIGHT = 120
        self.RECTANGLE_WIDTH = 1570
        self.TYPE_THRESHOLD = 40
        self.COLORMAP = np.array(
            [[28, 28, 28], [238, 44, 44], [250, 250, 250]])
        self.config = config
        self.output_dir = output_dir

        # There are 2 types of meters in test image datasets
        self.METER_CONFIG = self.config["meter_config"]
        self.SEG_LABEL = {'background': 0, 'pointer': 1, 'scale': 2}
        self.erode_kernel = 4
        self.score_threshold = 0.5
        self.seg_batch_size = 2
        self.input_shape = self.config["model_config"]["detector"]["input_shape"]

        ie_core = ov.Core()
        det_model_path = self.config["model_config"]["detector"]["model_path"]
        det_model_shape = self.config["model_config"]["detector"]["model_shape"]
        seg_model_path = self.config["model_config"]["segmenter"]["model_path"]
        seg_model_shape = self.config["model_config"]["segmenter"]["model_shape"]
        self.det_model = ie_core.read_model(det_model_path)
        self.det_model.reshape(det_model_shape)
        self.det_compiled_model = ie_core.compile_model(
            model=self.det_model, device_name=self.config["model_config"]["detector"]["device"])
        self.det_output_layer = self.det_compiled_model.output(0)

        self.seg_model = ie_core.read_model(seg_model_path)
        self.seg_model.reshape(seg_model_shape)
        self.seg_compiled_model = ie_core.compile_model(
            model=self.seg_model, device_name=self.config["model_config"]["segmenter"]["device"])
        self.seg_output_layer = self.seg_compiled_model.output(0)

    def postprocess(self, input):
        # Find the pointer location in scale map and calculate the meters reading
        rectangle_meters = self.circle_to_rectangle(input)
        line_scales, line_pointers = self.rectangle_to_line(rectangle_meters)
        binaried_scales = self.mean_binarization(line_scales)
        binaried_pointers = self.mean_binarization(line_pointers)
        scale_locations = self.locate_scale(binaried_scales)
        pointer_locations = self.locate_pointer(binaried_pointers)
        pointed_scales = self.get_relative_location(
            scale_locations, pointer_locations)
        meter_readings = self.calculate_reading(pointed_scales)

        # Plot the rectangle meters
        if len(rectangle_meters) == 2:
            rectangle_meters_stack = np.hstack([self.segmentation_map_to_image(rectangle_meters[0], self.COLORMAP),
                                                self.segmentation_map_to_image(rectangle_meters[1], self.COLORMAP)])
        else:
            rectangle_meters_stack = self.segmentation_map_to_image(
                rectangle_meters[0], self.COLORMAP)

        cv2.imwrite(os.path.join(self.output_dir, "rectangle_meters.jpg"),
                    cv2.cvtColor(rectangle_meters_stack, cv2.COLOR_RGB2BGR))

        return meter_readings

    def reading(self, input, image):
        # Create a final result photo with reading
        for i in range(len(input)):
            print("Meter {}: {:.3f}".format(i + 1, input[i]))

        result_image = image.copy()
        for i in range(len(self.loc)):
            cv2.rectangle(result_image, (self.loc[i][0], self.loc[i][1]), (
                self.loc[i][2], self.loc[i][3]), (0, 150, 0), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.rectangle(result_image, (self.loc[i][0], self.loc[i][1]), (
                self.loc[i][0] + 100, self.loc[i][1] + 40), (0, 150, 0), -1)
            cv2.putText(result_image, "#{:.3f}".format(
                input[i]), (self.loc[i][0], self.loc[i][1] + 25), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(self.output_dir, "reading_results.jpg"), result_image)

    def det_preprocess(self, input_image, target_size):
        """
        Preprocessing the input data for detection task

        Param: 
            input_image (np.array): input data
            size (int): the image size required by model input layer
        Retuns:
            img.astype (np.array): preprocessed image
        
        """
        if self.config["model_config"]["detector"]["color_format"] == "rgb":
            input_image = cv2.cvtColor(input_image, code=cv2.COLOR_BGR2RGB)
        img = cv2.resize(input_image, (target_size, target_size))
        img = np.transpose(img, [2, 0, 1]) / \
            self.config["model_config"]["detector"]["scale"]
        img = np.expand_dims(img, 0)
        img_mean = np.array(
            self.config["model_config"]["detector"]["mean"]).reshape((3, 1, 1))
        img_std = np.array(
            self.config["model_config"]["detector"]["std"]).reshape((3, 1, 1))
        img -= img_mean
        img /= img_std
        return img.astype(np.float32)

    def roi_process(self, input_images, target_size, interp=cv2.INTER_LINEAR):
        """
        Prepare the roi image of detection results data
        Preprocessing the input data for segmentation task

        Param：
            input_images (list[np.array])：the list of meter images
            target_size (list|tuple)： height and width of resized image， e.g [heigh,width]
            interp (int)：the interp method for image reszing

        Retuns：
            img_list (list[np.array])：the list of processed images
            resize_img (list[np.array]): for visualization
        
        """
        img_list = list()
        resize_list = list()
        for img in input_images:
            img_shape = img.shape
            scale_x = float(target_size[1]) / float(img_shape[1])
            scale_y = float(target_size[0]) / float(img_shape[0])
            resize_img = cv2.resize(
                img, None, None, fx=scale_x, fy=scale_y, interpolation=interp)
            resize_list.append(resize_img)
            if self.config["model_config"]["segmenter"]["color_format"] == "rgb":
                resize_img = cv2.cvtColor(resize_img, code=cv2.COLOR_BGR2RGB)
            resize_img = resize_img.transpose(2, 0, 1).astype(
                float) / self.config["model_config"]["segmenter"]["scale"]
            img_mean = np.array(
                self.config["model_config"]["segmenter"]["mean"]).reshape((3, 1, 1))
            img_std = np.array(
                self.config["model_config"]["segmenter"]["std"]).reshape((3, 1, 1))
            resize_img -= img_mean
            resize_img /= img_std
            img_list.append(resize_img)
        return img_list, resize_list

    def erode(self, seg_results, erode_kernel):
        """
        Erode the segmentation result to get the more clear instance of pointer and scale

        Param：
            seg_results (list[dict])：segmentation results
            erode_kernel (int): size of erode_kernel

        Return：
            eroded_results (list[dict])： the lab map of eroded_results
            
        """
        kernel = np.ones((erode_kernel, erode_kernel), np.uint8)
        eroded_results = seg_results
        for i in range(len(seg_results)):
            eroded_results[i] = cv2.erode(
                seg_results[i].astype(np.uint8), kernel)
        return eroded_results

    def circle_to_rectangle(self, seg_results):
        """
        Switch the shape of label_map from circle to rectangle

        Param：
            seg_results (list[dict])：segmentation results

        Return：
            rectangle_meters (list[np.array])：the rectangle of label map

        """
        rectangle_meters = list()
        for i, seg_result in enumerate(seg_results):
            label_map = seg_result

            # The size of rectangle_meter is determined by RECTANGLE_HEIGHT and RECTANGLE_WIDTH
            rectangle_meter = np.zeros(
                (self.RECTANGLE_HEIGHT, self.RECTANGLE_WIDTH), dtype=np.uint8)
            for row in range(self.RECTANGLE_HEIGHT):
                for col in range(self.RECTANGLE_WIDTH):
                    theta = self.PI * 2 * (col + 1) / self.RECTANGLE_WIDTH

                    # The radius of meter circle will be mapped to the height of rectangle image
                    rho = self.CIRCLE_RADIUS - row - 1
                    y = int(self.CIRCLE_CENTER[0] +
                            rho * math.cos(theta) + 0.5)
                    x = int(self.CIRCLE_CENTER[1] -
                            rho * math.sin(theta) + 0.5)
                    rectangle_meter[row, col] = label_map[y, x]
            rectangle_meters.append(rectangle_meter)
        return rectangle_meters

    def rectangle_to_line(self, rectangle_meters):
        """
        Switch the dimension of rectangle label map from 2D to 1D

        Param：
            rectangle_meters (list[np.array])：2D rectangle OF label_map。

        Return：
            line_scales (list[np.array])： the list of scales value
            line_pointers (list[np.array])：the list of pointers value

        """
        line_scales = list()
        line_pointers = list()
        for rectangle_meter in rectangle_meters:
            height, width = rectangle_meter.shape[0:2]
            line_scale = np.zeros((width), dtype=np.uint8)
            line_pointer = np.zeros((width), dtype=np.uint8)
            for col in range(width):
                for row in range(height):
                    if rectangle_meter[row, col] == self.SEG_LABEL['pointer']:
                        line_pointer[col] += 1
                    elif rectangle_meter[row, col] == self.SEG_LABEL['scale']:
                        line_scale[col] += 1
            line_scales.append(line_scale)
            line_pointers.append(line_pointer)
        return line_scales, line_pointers

    def mean_binarization(self, data_list):
        """
        Binarize the data

        Param：
            data_list (list[np.array])：input data

        Return：
            binaried_data_list (list[np.array])：output data。

        """
        batch_size = len(data_list)
        binaried_data_list = data_list
        for i in range(batch_size):
            mean_data = np.mean(data_list[i])
            width = data_list[i].shape[0]
            for col in range(width):
                if data_list[i][col] < mean_data:
                    binaried_data_list[i][col] = 0
                else:
                    binaried_data_list[i][col] = 1
        return binaried_data_list

    def locate_scale(self, line_scales):
        """
        Find location of center of each scale

        Param：
            line_scales (list[np.array])：the list of binaried scales value

        Return：
            scale_locations (list[list])：location of each scale

        """
        batch_size = len(line_scales)
        scale_locations = list()
        for i in range(batch_size):
            line_scale = line_scales[i]
            width = line_scale.shape[0]
            find_start = False
            one_scale_start = 0
            one_scale_end = 0
            locations = list()
            for j in range(width - 1):
                if line_scale[j] > 0 and line_scale[j + 1] > 0:
                    if not find_start:
                        one_scale_start = j
                        find_start = True
                if find_start:
                    if line_scale[j] == 0 and line_scale[j + 1] == 0:
                        one_scale_end = j - 1
                        one_scale_location = (
                            one_scale_start + one_scale_end) / 2
                        locations.append(one_scale_location)
                        one_scale_start = 0
                        one_scale_end = 0
                        find_start = False
            scale_locations.append(locations)
        return scale_locations

    def locate_pointer(self, line_pointers):
        """
        Find location of center of pointer

        Param：
            line_scales (list[np.array])：the list of binaried pointer value

        Return：
            scale_locations (list[list])：location of pointer

        """
        batch_size = len(line_pointers)
        pointer_locations = list()
        for i in range(batch_size):
            line_pointer = line_pointers[i]
            find_start = False
            pointer_start = 0
            pointer_end = 0
            location = 0
            width = line_pointer.shape[0]
            for j in range(width - 1):
                if line_pointer[j] > 0 and line_pointer[j + 1] > 0:
                    if not find_start:
                        pointer_start = j
                        find_start = True
                if find_start:
                    if line_pointer[j] == 0 and line_pointer[j + 1] == 0:
                        pointer_end = j - 1
                        location = (pointer_start + pointer_end) / 2
                        find_start = False
                        break
            pointer_locations.append(location)
        return pointer_locations

    def get_relative_location(self, scale_locations, pointer_locations):
        """
        Match location of pointer and scales

        Param：
            scale_locations (list[list])：location of each scale
            pointer_locations (list[list])：location of pointer

        Return：
            pointed_scales (list[dict])： a list of dict with:
                                        'num_scales': total number of scales
                                        'pointed_scale': predicted number of scales
                
        """
        pointed_scales = list()
        for scale_location, pointer_location in zip(scale_locations,
                                                    pointer_locations):
            num_scales = len(scale_location)
            pointed_scale = -1
            if num_scales > 0:
                for i in range(num_scales - 1):
                    if scale_location[i] <= pointer_location < scale_location[i + 1]:
                        pointed_scale = i + (pointer_location - scale_location[i]) / (
                            scale_location[i + 1] - scale_location[i] + 1e-05) + 1
            result = {'num_scales': num_scales, 'pointed_scale': pointed_scale}
            pointed_scales.append(result)
        return pointed_scales

    def calculate_reading(self, pointed_scales):
        """
        Calculate the value of meter according to the type of meter

        Param：
            pointed_scales (list[list])：predicted number of scales

        Return：
            readings (list[float])： the list of values read from meter
                
        """
        readings = list()
        batch_size = len(pointed_scales)
        for i in range(batch_size):
            pointed_scale = pointed_scales[i]
            # find the type of meter according the total number of scales
            if pointed_scale['num_scales'] > self.TYPE_THRESHOLD:
                reading = pointed_scale['pointed_scale'] * \
                    self.METER_CONFIG[0]['scale_interval_value']
            else:
                reading = pointed_scale['pointed_scale'] * \
                    self.METER_CONFIG[1]['scale_interval_value']
            readings.append(reading)
        return readings

    def segmentation_map_to_image(self,
                                  result: np.ndarray, colormap: np.ndarray, remove_holes: bool = False
                                  ) -> np.ndarray:
        """
        Convert network result of floating point numbers to an RGB image with
        integer values from 0-255 by applying a colormap.
        :param result: A single network result after converting to pixel values in H,W or 1,H,W shape.
        :param colormap: A numpy array of shape (num_classes, 3) with an RGB value per class.
        :param remove_holes: If True, remove holes in the segmentation result.
        :return: An RGB image where each pixel is an int8 value according to colormap.
        """
        if len(result.shape) != 2 and result.shape[0] != 1:
            raise ValueError(
                f"Expected result with shape (H,W) or (1,H,W), got result with shape {result.shape}"
            )

        if len(np.unique(result)) > colormap.shape[0]:
            raise ValueError(
                f"Expected max {colormap[0]} classes in result, got {len(np.unique(result))} "
                "different output values. Please make sure to convert the network output to "
                "pixel values before calling this function."
            )
        elif result.shape[0] == 1:
            result = result.squeeze(0)

        result = result.astype(np.uint8)

        contour_mode = cv2.RETR_EXTERNAL if remove_holes else cv2.RETR_TREE
        mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
        for label_index, color in enumerate(colormap):
            label_index_map = result == label_index
            label_index_map = label_index_map.astype(np.uint8) * 255
            contours, hierarchies = cv2.findContours(
                label_index_map, contour_mode, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(
                mask,
                contours,
                contourIdx=-1,
                color=color.tolist(),
                thickness=cv2.FILLED,
            )

        return mask
