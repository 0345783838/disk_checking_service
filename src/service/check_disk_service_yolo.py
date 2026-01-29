import math
import time
import cv2
import numpy as np
from src.dtos.meta import DataResponse, ErrorCode, DataDebugResponse, DataResponseUv
from src.service.base_service import BaseService
import base64
import ast


class DiskCheckingService(BaseService):
    def __init__(self):
        super().__init__()
        pass

    @staticmethod
    def _convert_2_base64(image):
        success, encoded_image = cv2.imencode('.png', image)
        if not success:
            return None
        image_bytes = encoded_image.tobytes()
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return img_base64

    @staticmethod
    def _get_box_centers(boxes):
        centers = []
        for box in boxes:
            x_center = (box[0] + box[2]) / 2
            y_center = (box[1] + box[3]) / 2
            centers.append([x_center, y_center])
        return np.array(centers)

    @staticmethod
    def euclidean_distance(pointA, pointB):
        return np.linalg.norm(pointA - pointB)

    @staticmethod
    def get_box_size(box):
        x1, y1, x2, y2 = box
        distance = x2 - x1 if x2 - x1 > y2 - y1 else y2 - y1
        return distance

    def draw_detected_boxes(self, image, boxes, confs, cls_idxs):
        for box, conf, cls in zip(boxes, confs, cls_idxs):
            xmin = int(box[0])
            ymin = int(box[1])
            xmax = int(box[2])
            ymax = int(box[3])
            object_conf = float(conf)

            # Draw the bounding box
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Prepare the label with class name and confidence
            label = f"{object_conf:.2f}"

            # Calculate the position for the label
            label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            top_left = (
            xmin, ymin - label_size[1] - 10 if ymin - label_size[1] - 10 > 10 else ymin + label_size[1] + 10)

            # Draw the label background
            cv2.rectangle(image, (top_left[0] - 1, top_left[1] + base_line + 10),
                          (top_left[0] + label_size[0], top_left[1] - label_size[1] + 10), (0, 255, 0), cv2.FILLED)

            # Put the label text
            cv2.putText(image, label, (top_left[0], top_left[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    def check_disk_debug(self, image, params):
        time_st = time.time()
        draw_image = image.copy()
        boxes, confs, cls_idxs = self.disk_point_detect_model.detect_objects_debug(image, params.detect_threshold,
                                                                                   params.detect_iou)
        if len(boxes) == 0:
            return

        # if len(boxes) < self.num_disk * 3:
        #     return

        # draw bounding box
        self.draw_detected_boxes(draw_image, boxes, confs, cls_idxs)
        res_detect = self._convert_2_base64(draw_image)

        # Groups the boxes by lines
        boxes_l1, boxes_l2, boxes_l3 = self.split_rows(boxes)

        # Align image by boxes
        crop_img, M, (w, h), quad_exp = self.full_rectify_pipeline(image, boxes_l1, boxes_l3, expand_ratio_x=0.2,
                                                                   expand_ratio_y=0.1)



        # Update all boxes coordinates to warped image
        boxes_l1 = self.update_boxes_after_warp(boxes_l1, M)
        boxes_l2 = self.update_boxes_after_warp(boxes_l2, M)
        boxes_l3 = self.update_boxes_after_warp(boxes_l3, M)

        # Get the point boxes by lines
        line_1_rects_bottom = self.get_line_boxes_ratio_shift(crop_img, boxes_l1, "bottom")
        line_2_rects_top = self.get_line_boxes_ratio_shift(crop_img, boxes_l2, "top")
        line_2_rects_bottom = self.get_line_boxes_ratio_shift(crop_img, boxes_l2, "bottom")
        line_3_rects_top = self.get_line_boxes_ratio_shift(crop_img, boxes_l3, "top")

        # Crop the boxes by lines
        line_1_crops_bottom = self.crop_boxes(crop_img, line_1_rects_bottom, "bottom")
        line_2_crops_top = self.crop_boxes(crop_img, line_2_rects_top, "top")
        line_2_crops_bottom = self.crop_boxes(crop_img, line_2_rects_bottom, "bottom")
        line_3_crops_top = self.crop_boxes(crop_img, line_3_rects_top, "top")
        print(f"Detect + Crop time: {(time.time() - time_st) * 1000:.2f} ms")

        # Classify the crops
        time_st = time.time()
        cls_res_l1_bottom, cls_conf_l1_bottom = self.point_classification_model.predict_batch(line_1_crops_bottom)
        cls_res_l2_top, cls_conf_l2_top = self.point_classification_model.predict_batch(line_2_crops_top)
        cls_res_l2_bottom, cls_conf_l2_bottom = self.point_classification_model.predict_batch(line_2_crops_bottom)
        cls_res_l3_bottom, cls_conf_l3_bottom = self.point_classification_model.predict_batch(line_3_crops_top)

        ng_boxes1 = [(box, conf) for label, box, conf in
                     zip(cls_res_l1_bottom, line_1_rects_bottom, cls_conf_l1_bottom)
                     if label == 'ng']
        ng_boxes2 = [(box, conf) for label, box, conf in zip(cls_res_l2_top, line_2_rects_top, cls_conf_l2_top) if
                     label == 'ng']
        ng_boxes3 = [(box, conf) for label, box, conf in
                     zip(cls_res_l2_bottom, line_2_rects_bottom, cls_conf_l2_bottom)
                     if label == 'ng']
        ng_boxes4 = [(box, conf) for label, box, conf in
                     zip(cls_res_l3_bottom, line_3_rects_top, cls_conf_l3_bottom) if
                     label == 'ng']

        # Merge near boxes
        ng_boxes1 = self.merge_boxes_1d_x(ng_boxes1)
        ng_boxes2 = self.merge_boxes_1d_x(ng_boxes2)
        ng_boxes3 = self.merge_boxes_1d_x(ng_boxes3)
        ng_boxes4 = self.merge_boxes_1d_x(ng_boxes4)

        ng_boxes = ng_boxes1 + ng_boxes2 + ng_boxes3 + ng_boxes4
        print(f"Classification time: {(time.time() - time_st) * 1000:.2f} ms")

        # Crop the segmentation area
        time_st = time.time()
        crop_seg_1, box_seg_1 = self.crop_box_for_segmentation(crop_img, boxes_l1[0], boxes_l2[0], ratio=0.35,
                                                    direction='bottom')
        crop_seg_2, box_seg_2 = self.crop_box_for_segmentation(crop_img, boxes_l2[0], boxes_l3[0])

        # Segment the disks using unet crop
        mask_seg_1, score_seg_1 = self.disk_segmentor_yolo.segment_large_image_debug(crop_seg_1, params.segment_threshold)
        mask_seg_2, score_seg_2 = self.disk_segmentor_yolo.segment_large_image_debug(crop_seg_2, params.segment_threshold)

        mask_seg_1 = self.clean_mask(mask_seg_1, params.disk_min_area)
        mask_seg_2 = self.clean_mask(mask_seg_2, params.disk_min_area)

        # Draw segment on crop image
        mask_crop = np.zeros((crop_img.shape[0], mask_seg_1.shape[1]), dtype=np.uint8)
        mask_crop[box_seg_1[1]:box_seg_1[3], box_seg_1[0]:box_seg_1[2]] = mask_seg_1
        mask_crop[box_seg_2[1]:box_seg_2[3], box_seg_2[0]:box_seg_2[2]] = mask_seg_2

        res_mark_crop = self._convert_2_base64(mask_crop)
        print(f"Segmentation time: {(time.time() - time_st) * 1000:.2f} ms")

        # Apply caliper
        time_st = time.time()
        center_1 = mask_seg_1.shape[1] // 2, int(mask_seg_1.shape[0] * 0.75)
        center_2 = mask_seg_1.shape[1] // 2, int(mask_seg_1.shape[0] * 0.25)
        center_3 = mask_seg_2.shape[1] // 2, mask_seg_2.shape[0] * 0.25
        center_4 = mask_seg_2.shape[1] // 2, mask_seg_2.shape[0] * 0.75
        caliper_res_1 = self.get_caliper_result_debug(mask_seg_1, center_1, params.caliper_length_rate, params.caliper_min_edge_distance,
                                                      params.caliper_max_edge_distance, params.caliper_thickness_list)
        caliper_res_2 = self.get_caliper_result_debug(mask_seg_1, center_2, params.caliper_length_rate, params.caliper_min_edge_distance,
                                                      params.caliper_max_edge_distance, params.caliper_thickness_list)
        caliper_res_3 = self.get_caliper_result_debug(mask_seg_2, center_3, params.caliper_length_rate, params.caliper_min_edge_distance,
                                                      params.caliper_max_edge_distance, params.caliper_thickness_list)
        caliper_res_4 = self.get_caliper_result_debug(mask_seg_2, center_4, params.caliper_length_rate, params.caliper_min_edge_distance,
                                                      params.caliper_max_edge_distance, params.caliper_thickness_list)
        print(f"Caliper time: {(time.time() - time_st) * 1000:.2f} ms")

        # Visualize result:
        self.draw_boxes(crop_img, ng_boxes, (0, 0, 255))
        self.draw_mask_contour(crop_seg_1, mask_seg_1, center_1)
        self.draw_mask_contour(crop_seg_1, mask_seg_1, center_2)
        self.draw_mask_contour(crop_seg_2, mask_seg_2, center_3)
        self.draw_mask_contour(crop_seg_2, mask_seg_2, center_4)
        res_spacing_1, dis_list_1, mids_1 = self.visualize_edge_spacing(crop_seg_1, caliper_res_1, params.disk_min_distance,
                                                    params.disk_max_distance)
        res_spacing_2, dis_list_2, mids_2 = self.visualize_edge_spacing(crop_seg_1, caliper_res_2, params.disk_min_distance,
                                                    params.disk_max_distance)
        res_spacing_3, dis_list_3, mids_3 = self.visualize_edge_spacing(crop_seg_2, caliper_res_3, params.disk_min_distance,
                                                    params.disk_max_distance)
        res_spacing_4, dis_list_4, mids_4 = self.visualize_edge_spacing(crop_seg_2, caliper_res_4, params.disk_min_distance,
                                                    params.disk_max_distance)

        res_final = self._convert_2_base64(crop_img)

        # Summary result
        res_classification = len(ng_boxes) == 0
        res_spacing = False not in res_spacing_1 + res_spacing_2 + res_spacing_3 + res_spacing_4
        res_count = (len(caliper_res_1["pairs"]) == self.num_disk and len(caliper_res_2["pairs"]) == self.num_disk
                     and len(caliper_res_3["pairs"]) == self.num_disk and len(caliper_res_4["pairs"]) == self.num_disk)

        sum_res = res_classification and res_spacing and res_count

        return DataDebugResponse(Result=sum_res,
                                 DetectImg=res_detect,
                                 SegmentImg=res_mark_crop,
                                 FinalImg=res_final)

    def check_disk_white(self, image):
        # return the image
        time_st = time.time()
        boxes, confs, cls_idxs = self.disk_point_detect_model(image)
        if len(boxes) == 0:
            # Return false
            return DataResponse(Result=False,
                                ErrorCode=ErrorCode.ERR_NUM_DISK[0],
                                ErrorDesc=ErrorCode.ERR_NUM_DISK[1])

        if len(boxes) < self.num_disk * 3:
            return DataResponse(Result=False,
                                ErrorCode=ErrorCode.ERR_NUM_DISK[0],
                                ErrorDesc=ErrorCode.ERR_NUM_DISK[1])

        # Groups the boxes by lines
        boxes_l1, boxes_l2, boxes_l3 = self.split_rows(boxes)

        # Align image by boxes
        crop_img, M, (w, h), quad_exp = self.full_rectify_pipeline(image, boxes_l1, boxes_l3, expand_ratio_x=0.2,
                                                                   expand_ratio_y=0.1)

        # Update all boxes coordinates to warped image
        boxes_l1 = self.update_boxes_after_warp(boxes_l1, M)
        boxes_l2 = self.update_boxes_after_warp(boxes_l2, M)
        boxes_l3 = self.update_boxes_after_warp(boxes_l3, M)

        # Get the coordinate for the UV image
        uv_box_l1 = self.get_uv_box(boxes_l1[0], w, M, "bottom")
        uv_box_l3 = self.get_uv_box(boxes_l3[0], w, M, "top")

        # Get the point boxes by lines
        line_1_rects_bottom = self.get_line_boxes_ratio_shift(crop_img, boxes_l1, "bottom")
        line_2_rects_top = self.get_line_boxes_ratio_shift(crop_img, boxes_l2, "top")
        line_2_rects_bottom = self.get_line_boxes_ratio_shift(crop_img, boxes_l2, "bottom")
        line_3_rects_top = self.get_line_boxes_ratio_shift(crop_img, boxes_l3, "top")

        # Crop the boxes by lines
        line_1_crops_bottom = self.crop_boxes(crop_img, line_1_rects_bottom, "bottom")
        line_2_crops_top = self.crop_boxes(crop_img, line_2_rects_top, "top")
        line_2_crops_bottom = self.crop_boxes(crop_img, line_2_rects_bottom, "bottom")
        line_3_crops_top = self.crop_boxes(crop_img, line_3_rects_top, "top")
        print(f"Detect + Crop time: {(time.time() - time_st) * 1000:.2f} ms")

        # Classify the crops
        time_st = time.time()
        cls_res_l1_bottom, cls_conf_l1_bottom = self.point_classification_model.predict_batch(line_1_crops_bottom)
        cls_res_l2_top, cls_conf_l2_top = self.point_classification_model.predict_batch(line_2_crops_top)
        cls_res_l2_bottom, cls_conf_l2_bottom = self.point_classification_model.predict_batch(line_2_crops_bottom)
        cls_res_l3_bottom, cls_conf_l3_bottom = self.point_classification_model.predict_batch(line_3_crops_top)

        ng_boxes1 = [(box, conf) for label, box, conf in zip(cls_res_l1_bottom, line_1_rects_bottom, cls_conf_l1_bottom)
                     if label == 'ng']
        ng_boxes2 = [(box, conf) for label, box, conf in zip(cls_res_l2_top, line_2_rects_top, cls_conf_l2_top) if
                     label == 'ng']
        ng_boxes3 = [(box, conf) for label, box, conf in zip(cls_res_l2_bottom, line_2_rects_bottom, cls_conf_l2_bottom)
                     if label == 'ng']
        ng_boxes4 = [(box, conf) for label, box, conf in zip(cls_res_l3_bottom, line_3_rects_top, cls_conf_l3_bottom) if
                     label == 'ng']

        # Merge near boxes
        ng_boxes1 = self.merge_boxes_1d_x(ng_boxes1)
        ng_boxes2 = self.merge_boxes_1d_x(ng_boxes2)
        ng_boxes3 = self.merge_boxes_1d_x(ng_boxes3)
        ng_boxes4 = self.merge_boxes_1d_x(ng_boxes4)

        ng_boxes = ng_boxes1 + ng_boxes2 + ng_boxes3 + ng_boxes4
        print(f"Classification time: {(time.time() - time_st) * 1000:.2f} ms")

        # Crop the segmentation area
        time_st = time.time()
        crop_seg_1,_ = self.crop_box_for_segmentation(crop_img, boxes_l1[0], boxes_l2[0], ratio=0.35, direction='bottom')
        crop_seg_2,_ = self.crop_box_for_segmentation(crop_img, boxes_l2[0], boxes_l3[0])

        # Segment the disks using unet crop
        # mask_seg_1, score_seg_1 = self.disk_segmentor(crop_seg_1)
        # mask_seg_2, score_seg_2 = self.disk_segmentor(crop_seg_2)

        mask_seg_1, _ = self.disk_segmentor_yolo(crop_seg_1)
        mask_seg_2, _ = self.disk_segmentor_yolo(crop_seg_2)

        mask_seg_1 = self.clean_mask(mask_seg_1, self.min_disk_area)
        mask_seg_2 = self.clean_mask(mask_seg_2, self.min_disk_area)

        print(f"Segmentation time: {(time.time() - time_st) * 1000:.2f} ms")

        # Apply caliper
        time_st = time.time()
        center_1 = mask_seg_1.shape[1] // 2, int(mask_seg_1.shape[0] * 0.75)
        center_2 = mask_seg_1.shape[1] // 2, int(mask_seg_1.shape[0] * 0.25)
        center_3 = mask_seg_2.shape[1] // 2, mask_seg_2.shape[0] * 0.25
        center_4 = mask_seg_2.shape[1] // 2, mask_seg_2.shape[0] * 0.75
        caliper_res_1 = self.get_caliper_result(mask_seg_1, center_1)
        caliper_res_2 = self.get_caliper_result(mask_seg_1, center_2)
        caliper_res_3 = self.get_caliper_result(mask_seg_2, center_3)
        caliper_res_4 = self.get_caliper_result(mask_seg_2, center_4)
        print(f"Caliper time: {(time.time() - time_st) * 1000:.2f} ms")

        # Visualize result:
        time_st = time.time()
        self.draw_boxes(crop_img, ng_boxes, (0, 0, 255))
        self.draw_mask_contour(crop_seg_1, mask_seg_1, center_1)
        self.draw_mask_contour(crop_seg_1, mask_seg_1, center_2)
        self.draw_mask_contour(crop_seg_2, mask_seg_2, center_3)
        self.draw_mask_contour(crop_seg_2, mask_seg_2, center_4)
        print(f"Draw time: {(time.time() - time_st) * 1000:.2f} ms")

        time_st = time.time()
        res_spacing_1, dis_list_1, mids_1 = self.visualize_edge_spacing(crop_seg_1, caliper_res_1, self.min_disk_distance,
                                                    self.max_disk_distance)
        res_spacing_2, dis_list_2, mids_2 = self.visualize_edge_spacing(crop_seg_1, caliper_res_2, self.min_disk_distance,
                                                    self.max_disk_distance)
        res_spacing_3, dis_list_3, mids_3 = self.visualize_edge_spacing(crop_seg_2, caliper_res_3, self.min_disk_distance,
                                                    self.max_disk_distance)
        res_spacing_4, dis_list_4, mids_4 = self.visualize_edge_spacing(crop_seg_2, caliper_res_4, self.min_disk_distance,
                                                    self.max_disk_distance)

        print(f"Spacing time: {(time.time() - time_st) * 1000:.2f} ms")

        time_st = time.time()
        # Summary result
        res_classification = len(ng_boxes) == 0
        res_spacing = False not in res_spacing_1 + res_spacing_2 + res_spacing_3 + res_spacing_4
        res_count = (len(caliper_res_1["pairs"]) == self.num_disk and len(caliper_res_2["pairs"]) == self.num_disk
                     and len(caliper_res_3["pairs"]) == self.num_disk and len(caliper_res_4["pairs"]) == self.num_disk)

        sum_res = res_classification and res_spacing and res_count
        min_disk_distance = min(dis_list_1 + dis_list_2 + dis_list_3 + dis_list_4)
        max_disk_distance = max(dis_list_1 + dis_list_2 + dis_list_3 + dis_list_4)

        print(f"Summary time: {(time.time() - time_st) * 1000:.2f} ms")

        if sum_res:
            return DataResponse(Result=sum_res,
                                ErrorCode=ErrorCode.PASS[0],
                                ErrorDesc=ErrorCode.PASS[1],
                                ResImg=self._convert_2_base64(crop_img),
                                MaxDiskDistance=max_disk_distance,
                                MinDiskDistance=min_disk_distance,
                                CropBox=str(quad_exp.tolist()),
                                UvBox1=str(uv_box_l1.tolist()),
                                UvBox2=str(uv_box_l3.tolist()),
                                Mid1=str(mids_1),
                                Mid2=str(mids_3),
                                )

        return DataResponse(Result=sum_res,
                            ErrorCode=ErrorCode.ABNORMAL[0],
                            ErrorDesc=ErrorCode.ABNORMAL[1],
                            ResImg=self._convert_2_base64(crop_img),
                            MaxDiskDistance=max_disk_distance,
                            MinDiskDistance=min_disk_distance,
                            CropBox=str(quad_exp.tolist()),
                            UvBox1=str(uv_box_l1.tolist()),
                            UvBox2=str(uv_box_l3.tolist()),
                            Mid1=str(mids_1),
                            Mid2=str(mids_3),
                            )

    def check_disk_swagger(self, image):
        # return the image
        time_st = time.time()
        boxes, confs, cls_idxs = self.disk_point_detect_model(image)
        if len(boxes) == 0:
            # Return false
            return DataResponse(Result=False)
        if len(boxes) < self.num_disk:
            return DataResponse(Result=False)

        # # draw bounding box
        # for box, conf, cls in zip(boxes, confs, cls_idxs):
        #     xmin = int(box[0])
        #     ymin = int(box[1])
        #     xmax = int(box[2])
        #     ymax = int(box[3])
        #     object_conf = float(conf)
        #
        #     # Draw the bounding box
        #     # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        #
        #     # Prepare the label with class name and confidence
        #     label = f"{object_conf:.2f}"
        #
        #     # Calculate the position for the label
        #     label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        #     top_left = (
        #     xmin, ymin - label_size[1] - 10 if ymin - label_size[1] - 10 > 10 else ymin + label_size[1] + 10)
        #
        #     # Draw the label background
        #     cv2.rectangle(image, (top_left[0] - 1, top_left[1] + base_line + 10),
        #                   (top_left[0] + label_size[0], top_left[1] - label_size[1] + 10), (0, 255, 0), cv2.FILLED)
        #
        #     # Put the label text
        #     cv2.putText(image, label, (top_left[0], top_left[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Groups the boxes by lines
        boxes_l1, boxes_l2, boxes_l3 = self.split_rows(boxes)

        # Align image by boxes
        crop_img, M, (w, h), quad_exp = self.full_rectify_pipeline(image, boxes_l1, boxes_l3, expand_ratio_x=0.2,
                                                                   expand_ratio_y=0.1)

        # Update all boxes coordinates to warped image
        boxes_l1 = self.update_boxes_after_warp(boxes_l1, M)
        boxes_l2 = self.update_boxes_after_warp(boxes_l2, M)
        boxes_l3 = self.update_boxes_after_warp(boxes_l3, M)

        # Get the point boxes by lines
        line_1_rects_bottom = self.get_line_boxes_ratio_shift(crop_img, boxes_l1, "bottom")
        line_2_rects_top = self.get_line_boxes_ratio_shift(crop_img, boxes_l2, "top")
        line_2_rects_bottom = self.get_line_boxes_ratio_shift(crop_img, boxes_l2, "bottom")
        line_3_rects_top = self.get_line_boxes_ratio_shift(crop_img, boxes_l3, "top")

        # Crop the boxes by lines
        line_1_crops_bottom = self.crop_boxes(crop_img, line_1_rects_bottom, "bottom")
        line_2_crops_top = self.crop_boxes(crop_img, line_2_rects_top, "top")
        line_2_crops_bottom = self.crop_boxes(crop_img, line_2_rects_bottom, "bottom")
        line_3_crops_top = self.crop_boxes(crop_img, line_3_rects_top, "top")
        print(f"Detect + Crop time: {(time.time() - time_st) * 1000:.2f} ms")

        # Classify the crops
        time_st = time.time()
        cls_res_l1_bottom, cls_conf_l1_bottom = self.point_classification_model.predict_batch(line_1_crops_bottom)
        cls_res_l2_top, cls_conf_l2_top = self.point_classification_model.predict_batch(line_2_crops_top)
        cls_res_l2_bottom, cls_conf_l2_bottom = self.point_classification_model.predict_batch(line_2_crops_bottom)
        cls_res_l3_bottom, cls_conf_l3_bottom = self.point_classification_model.predict_batch(line_3_crops_top)

        ng_boxes1 = [(box, conf) for label, box, conf in zip(cls_res_l1_bottom, line_1_rects_bottom, cls_conf_l1_bottom)
                     if label == 'ng']
        ng_boxes2 = [(box, conf) for label, box, conf in zip(cls_res_l2_top, line_2_rects_top, cls_conf_l2_top) if
                     label == 'ng']
        ng_boxes3 = [(box, conf) for label, box, conf in zip(cls_res_l2_bottom, line_2_rects_bottom, cls_conf_l2_bottom)
                     if label == 'ng']
        ng_boxes4 = [(box, conf) for label, box, conf in zip(cls_res_l3_bottom, line_3_rects_top, cls_conf_l3_bottom) if
                     label == 'ng']

        # Merge near boxes
        ng_boxes1 = self.merge_boxes_1d_x(ng_boxes1)
        ng_boxes2 = self.merge_boxes_1d_x(ng_boxes2)
        ng_boxes3 = self.merge_boxes_1d_x(ng_boxes3)
        ng_boxes4 = self.merge_boxes_1d_x(ng_boxes4)

        ng_boxes = ng_boxes1 + ng_boxes2 + ng_boxes3 + ng_boxes4
        print(f"Classification time: {(time.time() - time_st) * 1000:.2f} ms")

        # Crop the segmentation area
        time_st = time.time()
        crop_seg_1,_ = self.crop_box_for_segmentation(crop_img, boxes_l1[0], boxes_l2[0], ratio=0.35, direction='bottom')
        crop_seg_2,_ = self.crop_box_for_segmentation(crop_img, boxes_l2[0], boxes_l3[0])

        # Segment the disks using unet crop
        mask_seg_1, score_seg_1 = self.disk_segmentor_yolo(crop_seg_1)
        mask_seg_2, score_seg_2 = self.disk_segmentor_yolo(crop_seg_2)

        mask_seg_1 = self.clean_mask(mask_seg_1, self.min_disk_area)
        mask_seg_2 = self.clean_mask(mask_seg_2, self.min_disk_area)

        print(f"Segmentation time: {(time.time() - time_st) * 1000:.2f} ms")

        # Apply caliper
        time_st = time.time()
        center_1 = mask_seg_1.shape[1] // 2, int(mask_seg_1.shape[0] * 0.75)
        center_2 = mask_seg_1.shape[1] // 2, int(mask_seg_1.shape[0] * 0.25)
        center_3 = mask_seg_2.shape[1] // 2, mask_seg_2.shape[0] * 0.25
        center_4 = mask_seg_2.shape[1] // 2, mask_seg_2.shape[0] * 0.75
        caliper_res_1 = self.get_caliper_result(mask_seg_1, center_1)
        caliper_res_2 = self.get_caliper_result(mask_seg_1, center_2)
        caliper_res_3 = self.get_caliper_result(mask_seg_2, center_3)
        caliper_res_4 = self.get_caliper_result(mask_seg_2, center_4)
        print(f"Caliper time: {(time.time() - time_st) * 1000:.2f} ms")

        # Visualize result:
        self.draw_boxes(crop_img, ng_boxes, (0, 0, 255))
        self.draw_mask_contour(crop_seg_1, mask_seg_1, center_1)
        self.draw_mask_contour(crop_seg_1, mask_seg_1, center_2)
        self.draw_mask_contour(crop_seg_2, mask_seg_2, center_3)
        self.draw_mask_contour(crop_seg_2, mask_seg_2, center_4)

        self.visualize_edge_spacing(crop_seg_1, caliper_res_1, self.min_disk_distance,
                                                    self.max_disk_distance)
        self.visualize_edge_spacing(crop_seg_1, caliper_res_2, self.min_disk_distance,
                                                    self.max_disk_distance)
        self.visualize_edge_spacing(crop_seg_2, caliper_res_3, self.min_disk_distance,
                                                    self.max_disk_distance)
        self.visualize_edge_spacing(crop_seg_2, caliper_res_4, self.min_disk_distance,
                                                    self.max_disk_distance)

        return crop_img

    @staticmethod
    def visualize_edge_spacing(
            image: np.ndarray,
            caliper_result: dict,
            min_dist: float,
            max_dist: float,
            axis: int = 0,
            line_thickness: int = 2,
            font_scale: float = 0.45,
            midpoint_radius: int = 7
    ):

        # 1. compute midpoints
        mids = []
        distance_list = []
        for pair in caliper_result["pairs"]:
            p1 = np.array(pair["e1"]["point"], dtype=np.float32)
            p2 = np.array(pair["e2"]["point"], dtype=np.float32)
            mids.append((p1 + p2) / 2.0)

        # 2. sort midpoints
        mids = sorted(mids, key=lambda m: m[axis])

        results = []

        # 3. draw midpoints (ORANGE)
        for m in mids:
            cv2.circle(
                image,
                tuple(m.astype(int)),
                midpoint_radius,
                (0, 127, 255),  # orange
                -1
            )

        # 4. draw spacing + text AT TRUE CENTER
        for i in range(len(mids) - 1):
            m1 = mids[i]
            m2 = mids[i + 1]

            dist = float(np.linalg.norm(m2 - m1))
            distance_list.append(dist)
            is_ng = dist < min_dist or dist > max_dist
            color = (0, 0, 255) if is_ng else (0, 255, 0)

            p1 = tuple(m1.astype(int))
            p2 = tuple(m2.astype(int))

            # line between edges
            cv2.line(image, p1, p2, color, line_thickness)

            # TRUE center between two edges
            mid_text = ((m1 + m2) / 2).astype(int)
            label = f"{dist:.1f}"

            # center text exactly
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            cv2.putText(
                image,
                label,
                (mid_text[0] - tw // 2, mid_text[1] - th // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                1,
                cv2.LINE_AA
            )

            results.append(False if is_ng else True)

        # 5. re - draw midpoints (ORANGE)
        for m in mids:
            cv2.circle(
                image,
                tuple(m.astype(int)),
                1,
                (0, 127, 255),  # orange
                -1
            )

        return results, distance_list, [x.tolist() for x in mids]

    @staticmethod
    def visualize_edge_spacing_uv(
            image: np.ndarray,
            center: tuple,
            caliper_result: dict,
            axis: int = 0,
            line_thickness: int = 2,
            midpoint_radius: int = 7,

    ):

        # 1. compute midpoints
        mids = []
        for pair in caliper_result["pairs"]:
            p1 = np.array(pair["e1"]["point"], dtype=np.float32)
            p2 = np.array(pair["e2"]["point"], dtype=np.float32)
            mids.append((p1 + p2) / 2.0)

        # 2. sort midpoints
        mids = sorted(mids, key=lambda m: m[axis])
        # 3. Draw center line
        cv2.line(image, (0, center[1]), (image.shape[1], center[1]), (0, 255, 0), line_thickness)
        # 4. draw midpoints (ORANGE)
        for m in mids:
            cv2.circle(
                image,
                tuple(m.astype(int)),
                midpoint_radius,
                (247, 192, 27),  # orange
                -1
            )

        return image, mids

    @staticmethod
    def merge_boxes_1d_x(box_score_list, x_gap=0):
        """
        Merge boxes assuming they lie on the same horizontal line (Y nearly equal)

        x_gap: cho phép khoảng hở nhỏ giữa các box vẫn được merge
        """
        if len(box_score_list) == 0:
            return []

        boxes = [b for b, _ in box_score_list]

        # sort theo x1
        boxes = sorted(boxes, key=lambda b: b[0])

        merged = []
        cur = boxes[0]

        for box in boxes[1:]:
            # nếu overlap hoặc chạm theo trục X
            if box[0] <= cur[2] + x_gap:
                cur = [
                    min(cur[0], box[0]),
                    min(cur[1], box[1]),
                    max(cur[2], box[2]),
                    max(cur[3], box[3]),
                ]
            else:
                merged.append(cur)
                cur = box

        merged.append(cur)
        return merged

    @staticmethod
    def draw_stripes_on_contour_inplace(
            image, contour,
            stripe_spacing=10,
            color=(0, 255, 0),
            thickness=1
    ):
        if len(contour) == 0:
            return

        cnt = contour[0]

        x, y, w, h = cv2.boundingRect(cnt)

        if w <= 0 or h <= 0:
            return

        roi = image[y:y + h, x:x + w]

        # shift contour về local ROI
        cnt_local = cnt.copy()
        cnt_local[:, 0, 0] -= x
        cnt_local[:, 0, 1] -= y

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [cnt_local], -1, 255, -1)

        stripe_layer = np.zeros_like(roi)

        for i in range(-h, w, stripe_spacing):
            cv2.line(
                stripe_layer,
                (i, h),
                (i + h, 0),
                color,
                thickness
            )

        m = mask > 0
        roi[m] = (
                roi[m].astype(np.float32) * 0.9 +
                stripe_layer[m].astype(np.float32) * 0.9
        ).clip(0, 255).astype(np.uint8)

    @staticmethod
    def draw_boxes(image, boxes, color):
        for box in boxes:
            if type(box) == tuple:
                bb = box[0]
                score = box[1]
                cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), color, 1)
                cv2.putText(image, f"{score:.2f}", (int(bb[0]), int(bb[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                box_height = box[3] - box[1]
                new_y1 = int(box[1] + box_height * 0.2)
                new_y2 = int(box[3] - box_height * 0.2)
                offset_x = 4
                offset_y = 2
                (tw, th), _ = cv2.getTextSize(f"{'NG'}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, (int(box[0]), new_y1), (int(box[2]), new_y2), color, 2)

                cv2.rectangle(image, (int(box[0]), new_y1 - th - offset_y), (int(box[0]) + tw + offset_x, new_y1),
                              color, cv2.FILLED)
                cv2.putText(image, f"{'NG'}", (int(box[0]) + offset_x // 2, new_y1 - offset_y // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    @staticmethod
    def order_quad_pts(pts):
        """
        Input: pts shape (4,2) unsorted
        Output: ordered as [tl, tr, br, bl]
        """
        pts = np.array(pts, dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    # @staticmethod
    # def expand_quad_towards_center(quad, ratio=0.10):
    #     """
    #     Expand quad points away from center by ratio.
    #     quad: (4,2)
    #     """
    #     center = np.mean(quad, axis=0)
    #     return center + (quad - center) * (1.0 + ratio)

    @staticmethod
    def expand_quad_towards_center(quad, ratio_x=0.10, ratio_y=0.10):
        """
        Expand quad points away from center by ratio.
        quad: (4,2)
        """
        quad = np.asarray(quad, dtype=np.float32)
        center = np.mean(quad, axis=0)  # (cx, cy)

        # scale matrix for x,y
        scale = np.array([1.0 + ratio_x, 1.0 + ratio_y], dtype=np.float32)

        # (quad - center) * scale + center
        return center + (quad - center) * scale

    @staticmethod
    def quad_to_rect_size(quad):
        """
        Compute destination rectangle width and height from ordered quad (tl,tr,br,bl)
        Use max of top/bottom edge lengths for width, left/right edges for height.
        """
        tl, tr, br, bl = quad
        width_top = np.linalg.norm(tr - tl)
        width_bot = np.linalg.norm(br - bl)
        max_w = int(round(max(width_top, width_bot)))
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_h = int(round(max(height_left, height_right)))
        # Avoid zero
        max_w = max(1, max_w)
        max_h = max(1, max_h)
        return max_w, max_h

    def warp_quad_to_rect(self, image, quad, expand_ratio_x=0.10, expand_ratio_y=0.10,
                          border_mode=cv2.BORDER_REPLICATE):
        """
        quad: (4,2) unsorted
        returns: warped_img, M (3x3), dst_bbox = (w,h)
        """
        # order
        quad_ord = self.order_quad_pts(quad)
        # expand
        quad_exp = self.expand_quad_towards_center(quad_ord, ratio_x=expand_ratio_x, ratio_y=expand_ratio_y)
        # destination size
        w, h = self.quad_to_rect_size(quad_exp)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        # compute homography
        M = cv2.getPerspectiveTransform(quad_exp.astype(np.float32), dst)
        warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=border_mode)
        return warped, M, (w, h), quad_exp

    @staticmethod
    def transform_points(M, pts):
        """
        Apply homography M to pts (N,2) and return transformed (N,2)
        """
        pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float32)])  # (N,3)
        dst_h = (M @ pts_h.T).T  # (N,3)
        dst = dst_h[:, :2] / dst_h[:, 2:3]
        return dst

    def update_boxes_after_warp(self, boxes, M):
        """
        boxes: (N,4) as [x1,y1,x2,y2] axis-aligned in original image
        M: homography from original -> warped
        returns: new_boxes (N,4) in warped image coords (axis-aligned)
        """
        new_boxes = []
        for b in boxes:
            x1, y1, x2, y2 = b
            # four corners
            corners = np.array([
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ], dtype=np.float32)
            t = self.transform_points(M, corners)  # (4,2)
            # axis aligned bbox in warped image
            x_min = float(np.min(t[:, 0]))
            y_min = float(np.min(t[:, 1]))
            x_max = float(np.max(t[:, 0]))
            y_max = float(np.max(t[:, 1]))
            new_boxes.append([x_min, y_min, x_max, y_max])
        return np.array(new_boxes, dtype=np.float32)

    # ---------- Example usage ----------
    # image: original image
    # row1, row3: arrays of boxes for top and bottom rows (format [x1,y1,x2,y2])
    # boxes: original full list of boxes (N,4)

    def full_rectify_pipeline(self, image, row1, row3, expand_ratio_x=0.10, expand_ratio_y=0.10):
        # select P1-P4 as user described
        left_top_box = row1[np.argmin(row1[:, 0])]
        right_top_box = row1[np.argmax(row1[:, 2])]
        left_bot_box = row3[np.argmin(row3[:, 0])]
        right_bot_box = row3[np.argmax(row3[:, 2])]

        P1 = [left_top_box[0], left_top_box[1]]  # top-left
        P2 = [right_top_box[2], right_top_box[1]]  # top-right (use x2,y1)
        P3 = [right_bot_box[2], right_bot_box[3]]  # bottom-right (x2,y2)
        P4 = [left_bot_box[0], left_bot_box[3]]  # bottom-left (x1,y2)

        quad = np.array([P1, P2, P3, P4], dtype=np.float32)

        warped, M, (w, h), quad_exp = self.warp_quad_to_rect(image, quad, expand_ratio_x=expand_ratio_x,
                                                             expand_ratio_y=expand_ratio_y)
        # Update all boxes coordinates to warped image

        return warped, M, (w, h), quad_exp

    @staticmethod
    def split_rows(boxes):
        # Tính tâm Y
        cy = (boxes[:, 1] + boxes[:, 3]) / 2

        # Sort theo Y
        idx_sorted = np.argsort(cy)
        boxes_sorted = boxes[idx_sorted]
        cy_sorted = cy[idx_sorted]

        # Tính khoảng cách giữa các tâm kế nhau
        diffs = np.diff(cy_sorted)

        # Tìm 2 vị trí jump lớn nhất → ngăn thành 3 hàng
        jump_idx = np.argsort(diffs)[-2:]  # 2 bước nhảy lớn nhất
        jump_idx = np.sort(jump_idx)

        # Chia thành 3 nhóm
        r1 = boxes_sorted[:jump_idx[0] + 1]
        r2 = boxes_sorted[jump_idx[0] + 1: jump_idx[1] + 1]
        r3 = boxes_sorted[jump_idx[1] + 1:]

        return r1, r2, r3

    @staticmethod
    def get_ratio_shift_box(
            image,
            box,
            direction="bottom",
            ratio_width=1.0,
            ratio_height=0.5,
            ratio_shift_y=0.0,

    ):
        """
        box = [x1, y1, x2, y2]
        direction: 'top' hoặc 'bottom'
        ratio_height: phần trăm chiều cao muốn crop (0→1)
        ratio_shift_y: phần trăm dịch tâm theo chiều cao box (0→1)
        ratio_width: mở rộng crop theo chiều rộng (1.0 = giữ nguyên, 2.0 = gấp đôi)
        """

        x1, y1, x2, y2 = map(int, box)

        h = y2 - y1
        w = x2 - x1

        # --- Height crop ---
        crop_h = int(h * ratio_height)

        # center Y
        cy = (y1 + y2) / 2.0
        shift_y = h * ratio_shift_y

        if direction == "bottom":
            cy_shifted = cy + shift_y
        elif direction == "top":
            cy_shifted = cy - shift_y
        else:
            raise ValueError("direction must be 'top' or 'bottom'")

        y1_new = int(cy_shifted - crop_h / 2)
        y2_new = int(y1_new + crop_h)

        # --- Width crop ---
        new_w = w * ratio_width
        cx = (x1 + x2) / 2.0

        x1_new = int(cx - new_w / 2)
        x2_new = int(cx + new_w / 2)

        # Không clamp: numpy tự cắt nếu vượt ảnh
        return [max(x1_new, 0), max(y1_new, 0), max(x2_new, 0), max(y2_new, 0)]

    def get_line_boxes_ratio_shift(self, image, line_boxes, direction, ratio_width=1.5, ratio_height=1.3,
                                   ratio_shift_y=0.5):
        boxes = []
        for box in line_boxes:
            new_box = self.get_ratio_shift_box(image, box, direction, ratio_width, ratio_height, ratio_shift_y)
            boxes.append(new_box)
        return boxes

    @staticmethod
    def crop_boxes(image, boxes, direction):
        crops = []
        if direction == 'top':
            for box in boxes:
                crop = image[box[1]:box[3], box[0]:box[2]]
                crop = cv2.rotate(crop, cv2.ROTATE_180)
                crops.append(crop)
        else:
            for box in boxes:
                crop = image[box[1]:box[3], box[0]:box[2]]
                crops.append(crop)
        return crops

    @staticmethod
    def crop_box_for_segmentation(image, box_line_1, box_line_2, ratio=0.35, direction='top'):
        y_1 = int(box_line_1[3])
        y_2 = int(box_line_2[1])
        ratio_height = int((y_2 - y_1) * ratio)
        if direction == 'top':
            box = [0, y_1, image.shape[1], y_1 + ratio_height]
        else:
            box = [0, y_2 - ratio_height, image.shape[1], y_2]

        crop = image[box[1]:box[3], box[0]:box[2]]
        return crop, box

    def get_caliper_result(self, img, center):
        res = self.caliper.measure(img, center=center)
        # vis = self.caliper.visualize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), center=center)
        return res

    def clean_mask(self, img, min_disk_area):
        img = self.remove_mask_noise(img, min_disk_area)
        # img_close = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((7, 3)))

        return img

    def get_caliper_result_debug(self, img, center, length_rate, min_edge_distance, max_edge_distance, thickness_list):
        # img_open = cv2.morphologyEx(img_close, cv2.MORPH_OPEN, np.ones((1, 3)))

        res = self.caliper.measure_debug(img,
                                         center=center,
                                         min_edge_distance=min_edge_distance,
                                         max_edge_distance=max_edge_distance,
                                         length_rate=length_rate,
                                         thickness_list=thickness_list)

        # vis = self.caliper.visualize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), center=center)
        return res

    def draw_mask_contour(self, img, mask_seg, center, draw_ratio=0.4, mask_color=(255, 153, 51),
                          cnt_color=(255, 102, 0)):
        x, y = center
        img_h, img_w, _ = img.shape

        y1 = max(0, int(y - img_h * draw_ratio / 2.0))
        y2 = min(img_h, int(y + img_h * draw_ratio / 2.0))

        mask_seg_center = mask_seg.copy()
        mask_seg_center[0:y1, :] = 0
        mask_seg_center[y2:, :] = 0

        contours, _ = cv2.findContours(mask_seg_center, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            cv2.drawContours(img, [contour], -1, cnt_color, 1)
            self.draw_stripes_on_contour_inplace(img, [contour], color=mask_color)

    def remove_mask_noise(self, img, min_disk_area):
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_disk_area:
                cv2.drawContours(img, [contour], -1, 0, -1)

        return img

    def check_disk_uv(self, img, crop_box, uv_box_1, uv_box_2, mid_1, mid_2):
        crop_box = np.array(ast.literal_eval(crop_box), dtype=np.float32)
        uv_box_1 = np.array(ast.literal_eval(uv_box_1), dtype=np.int32)
        uv_box_2 = np.array(ast.literal_eval(uv_box_2), dtype=np.int32)
        mid_1 = np.array(ast.literal_eval(mid_1), dtype=np.float32)
        mid_2 = np.array(ast.literal_eval(mid_2), dtype=np.float32)

        crop_img = self.crop_by_4pts(img, crop_box)

        uv_box_1 = np.array(uv_box_1, dtype=np.int32)
        uv_box_2 = np.array(uv_box_2, dtype=np.int32)

        uv_crop_1 = crop_img[uv_box_1[0][1]:uv_box_1[2][1], uv_box_1[0][0]:uv_box_1[2][0]]
        uv_crop_2 = crop_img[uv_box_2[0][1]:uv_box_2[2][1], uv_box_2[0][0]:uv_box_2[2][0]]

        uv_thresh_1 = self.preprocess_uv_image(uv_crop_1, threshold=self.uv_disk_threshold)
        uv_thresh_2 = self.preprocess_uv_image(uv_crop_2, threshold=self.uv_disk_threshold)

        uv_thresh_1 = self.remove_mask_noise(uv_thresh_1, min_disk_area=self.uv_min_disk_area)
        uv_thresh_2 = self.remove_mask_noise(uv_thresh_2, min_disk_area=self.uv_min_disk_area)

        caliper_res_1 = self.get_caliper_result_debug(uv_thresh_1, center=(uv_crop_1.shape[1] // 2, uv_crop_1.shape[0] // 2),
                                                      length_rate=0.95,
                                                      max_edge_distance=50,
                                                      min_edge_distance=5,
                                                      thickness_list=[1, 3])

        caliper_res_2 = self.get_caliper_result_debug(uv_thresh_2, center=(uv_crop_2.shape[1] // 2, uv_crop_2.shape[0] // 2),
                                                      length_rate=self.caliper.length_rate,
                                                      max_edge_distance=self.caliper.pair_max_gap,
                                                      min_edge_distance=self.caliper.min_edge_distance,
                                                      thickness_list=self.caliper.thickness_list
                                                      )

        uv_crop_1 = self.draw_uv_mask(uv_crop_1, uv_thresh_1)
        uv_crop_2 = self.draw_uv_mask(uv_crop_2, uv_thresh_2)

        result_1, mid_uv_1 = self.visualize_edge_spacing_uv(uv_crop_1,  (uv_crop_1.shape[1] // 2, uv_crop_1.shape[0] // 2), caliper_res_1)
        result_2, mid_uv_2 = self.visualize_edge_spacing_uv(uv_crop_2, (uv_crop_2.shape[1] // 2, uv_crop_2.shape[0] // 2), caliper_res_2)

        self.draw_uv_index(uv_crop_1, mid_uv_1, mid_1)
        self.draw_uv_index(uv_crop_2, mid_uv_2, mid_2)


        # Draw segment on crop image
        mask_crop = crop_img.copy()
        # mask_crop = np.zeros((crop_img.shape[0], crop_img.shape[1], 3), dtype=np.uint8)
        mask_crop[uv_box_1[0][1]:uv_box_1[2][1], uv_box_1[0][0]:uv_box_1[2][0]] = result_1
        mask_crop[uv_box_2[0][1]:uv_box_2[2][1], uv_box_2[0][0]:uv_box_2[2][0]] = result_2

        count_uv_disk = len(caliper_res_1["pairs"]) + len(caliper_res_2["pairs"])

        if count_uv_disk == 0:
            return DataResponseUv(Result=True,
                                  CountUvDisk=0,
                                  ErrorCode=ErrorCode.PASS[0],
                                  ErrorDesc=ErrorCode.PASS[1],
                                  ResImg=self._convert_2_base64(mask_crop))

        return DataResponseUv(Result=False,
                              CountUvDisk=count_uv_disk,
                              ErrorCode=ErrorCode.ERR_NUM_UV_DISK[0],
                              ErrorDesc=ErrorCode.ERR_NUM_UV_DISK[1],
                              ResImg=self._convert_2_base64(mask_crop))

    @staticmethod
    def preprocess_uv_image(image, threshold):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray, 5)
        _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

        return thresh

    @staticmethod
    def crop_by_4pts(image, pts):
        """
        image: np.ndarray (H, W, C)
        pts: array-like shape (4, 2), 4 điểm bất kỳ trên ảnh gốc

        return:
            cropped_img: ảnh đã crop + align
            M: perspective transform matrix
        """
        pts = np.array(pts, dtype=np.float32)

        # --- 1. Sắp xếp 4 điểm theo thứ tự: tl, tr, br, bl ---
        def order_points(pts):
            rect = np.zeros((4, 2), dtype=np.float32)

            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)

            rect[0] = pts[np.argmin(s)]  # top-left
            rect[2] = pts[np.argmax(s)]  # bottom-right
            rect[1] = pts[np.argmin(diff)]  # top-right
            rect[3] = pts[np.argmax(diff)]  # bottom-left

            return rect

        rect = order_points(pts)

        # --- 2. Tính width / height output ---
        w1 = np.linalg.norm(rect[1] - rect[0])
        w2 = np.linalg.norm(rect[2] - rect[3])
        width = int(max(w1, w2))

        h1 = np.linalg.norm(rect[3] - rect[0])
        h2 = np.linalg.norm(rect[2] - rect[1])
        height = int(max(h1, h2))

        # --- 3. Điểm đích ---
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)

        # --- 4. Perspective transform ---
        M = cv2.getPerspectiveTransform(rect, dst)
        cropped = cv2.warpPerspective(
            image, M, (width, height),
            flags=cv2.INTER_LINEAR
        )

        return cropped

    def get_uv_box(self, box, w, M, direction, ratio_h=2,):
        x1, y1, x2, y2 = box
        height = y2 - y1
        if direction == "bottom":
            y1_uv = y2 + 0.5*height
            y2_uv = y1_uv + height*ratio_h + 0.5*height

            pts_warped = np.array([
                [0, y1_uv],
                [w, y1_uv],
                [w, y2_uv],
                [0, y2_uv]
            ], dtype=np.float32)

            # pts_warped = pts_warped.reshape(-1, 1, 2)
            # M_inv = np.linalg.inv(M)
            # pts_image = cv2.perspectiveTransform(pts_warped, M_inv)
            # pts_image = pts_image.reshape(-1, 2)

        else:
            y1_uv = y1 - height*ratio_h - 0.5*height
            y2_uv = y1 - 0.5*height

            pts_warped = np.array([
                [0, y1_uv],
                [w, y1_uv],
                [w, y2_uv],
                [0, y2_uv]
            ], dtype=np.float32)

            # pts_warped = pts_warped.reshape(-1, 1, 2)
            # M_inv = np.linalg.inv(M)
            # pts_image = cv2.perspectiveTransform(pts_warped, M_inv)
            # pts_image = pts_image.reshape(-1, 2)

        return pts_warped

    def draw_uv_mask(self, uv_crop_1, uv_thresh_1, cnt_color=(0, 0, 255), mask_color=(0, 255, 0)):
        contours, _ = cv2.findContours(uv_thresh_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            cv2.drawContours(uv_crop_1, [contour], -1, cnt_color, 1)
            self.draw_stripes_on_contour_inplace(uv_crop_1, [contour], color=mask_color)

        return uv_crop_1

    def draw_uv_index(self, uv_crop_1, mid_uv_1, mid_1):
        if len(mid_uv_1) < 1 or len(mid_1) < 1:
            return

        query = np.array(mid_uv_1)
        x_pts = mid_1[:, 0]  # (N,)
        x_q = query[:, 0]  # (M,)

        # Tính |x_q - x_pts| cho toàn bộ cặp (broadcast)
        dx = np.abs(x_q[:, None] - x_pts[None, :])  # (M,N)

        # Lấy index pts gần nhất cho từng query
        idxs = np.argmin(dx, axis=1)   # (M,)

        # vẽ số index lên điểm các điểm trong mid_1
        for i, pt in enumerate(mid_uv_1):
            cv2.putText(uv_crop_1, f"disk_{idxs[i]+1}", (int(pt[0]) - 30, int(pt[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    def check_disk_uv_debug(self, img, params):
        pass


if __name__ == '__main__':
    import glob
    from tqdm import tqdm
    import os

    IMAGE_PATH = r"D:\huynhvc\OTHERS\disk_checking\disk_checking\raw_data\08_12\conlai"
    OUTPUT_PATH = r"D:\huynhvc\OTHERS\disk_checking\disk_checking\datasets\dataset_cls\working\out_rect"
    save_path_bottom_rect = f"{OUTPUT_PATH}/bottom"
    save_path_top_rect = f"{OUTPUT_PATH}/top"
    os.makedirs(save_path_bottom_rect, exist_ok=True)
    os.makedirs(save_path_top_rect, exist_ok=True)

    disk_checking_service = DiskCheckingService()

    paths = glob.glob(f"{IMAGE_PATH}/*")

    for path in tqdm(paths):
        image = cv2.imread(path)
        boxes, confs, cls_idxs = disk_checking_service.disk_point_detect_model(image)

        # GET THE CLASSIFICATION BOXES
        # Groups the boxes by lines
        boxes_l1, boxes_l2, boxes_l3 = disk_checking_service.split_rows(boxes)

        # Align image by boxes
        crop_img, M, (w, h), quad_exp = disk_checking_service.full_rectify_pipeline(image, boxes_l1, boxes_l3,
                                                                                    expand_ratio_x=0.2,
                                                                                    expand_ratio_y=0.1)

        # Update all boxes coordinates to warped image
        boxes_l1 = disk_checking_service.update_boxes_after_warp(boxes_l1, M)
        boxes_l2 = disk_checking_service.update_boxes_after_warp(boxes_l2, M)
        boxes_l3 = disk_checking_service.update_boxes_after_warp(boxes_l3, M)

        # # Get the point boxes by lines
        # line_1_rects_bottom = disk_checking_service.get_line_boxes_ratio_shift(crop_img, boxes_l1, "bottom")
        # line_2_rects_top = disk_checking_service.get_line_boxes_ratio_shift(crop_img, boxes_l2, "top")
        # line_2_rects_bottom = disk_checking_service.get_line_boxes_ratio_shift(crop_img, boxes_l2, "bottom")
        # line_3_rects_top = disk_checking_service.get_line_boxes_ratio_shift(crop_img, boxes_l3, "top")
        #
        # # Crop the boxes by lines
        # line_1_crops_bottom = disk_checking_service.crop_boxes(crop_img, line_1_rects_bottom, "bottom")
        # line_2_crops_top = disk_checking_service.crop_boxes(crop_img, line_2_rects_top, "top")
        # line_2_crops_bottom = disk_checking_service.crop_boxes(crop_img, line_2_rects_bottom, "bottom")
        # line_3_crops_top = disk_checking_service.crop_boxes(crop_img, line_3_rects_top, "top")
        #
        # for i, crop_rect in enumerate(line_1_crops_bottom + line_2_crops_bottom):
        #     img_name = os.path.basename(path).replace('.bmp', f'_{i}.bmp')
        #     # cv2.imwrite(fr"{save_path_bottom_rect}/{img_name}", crop_rect)
        #
        # for j, crop_rect in enumerate(line_2_crops_top + line_3_crops_top):
        #     img_name = os.path.basename(path).replace('.bmp', f'_{i+j+1}.bmp')
        #     cv2.imwrite(fr"{save_path_top_rect}/{img_name}", crop_rect)

        # --- GET THE CROPS FOR SEGMENTATION
        crop_seg_1 = disk_checking_service.crop_box_for_segmentation(crop_img, boxes_l1[0], boxes_l2[0], ratio=0.35,
                                                                     direction='bottom')
        crop_seg_2 = disk_checking_service.crop_box_for_segmentation(crop_img, boxes_l2[0], boxes_l3[0])


        # crop boxes
        def crop_images(image):
            H, W = image.shape[:2]
            step = W // 6
            overlap = int(step * 0.05)

            crops = []
            positions = []

            x = 0
            while x < W and len(crops) < 6:
                x_start = max(0, x - overlap)
                x_end = min(W, x + step + overlap)

                crop = image[:, x_start:x_end]
                crops.append(crop)
                positions.append((x_start, x_end))

                x += step

            return crops


        crops_1 = crop_images(crop_seg_1)
        crops_2 = crop_images(crop_seg_2)

        for i, crop in enumerate(crops_1 + crops_2):
            img_name = os.path.basename(path).replace('.bmp', f'_crop_{i}.bmp')
            cv2.imwrite(fr"D:\huynhvc\OTHERS\disk_checking\disk_checking\testing\out_rect_segment/images/{img_name}",
                        crop)

        # img_name_1 = os.path.basename(path).replace('.bmp', f'_seg_1.bmp')
        # img_name_2 = os.path.basename(path).replace('.bmp', f'_seg_2.bmp')
        # cv2.imwrite(fr"D:\huynhvc\OTHERS\disk_checking\disk_checking\testing\out_rect_segment/{img_name_1}", crop_seg_1)
        # cv2.imwrite(fr"D:\huynhvc\OTHERS\disk_checking\disk_checking\testing\out_rect_segment/{img_name_2}", crop_seg_2)

        # Segmentation
