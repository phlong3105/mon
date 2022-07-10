#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm

import coco_tools

categories = np.array([
	{"id": 0, "name": "human"},
	{"id": 1, "name": "bicycle"},
	{"id": 2, "name": "motorcycle"},
	{"id": 3, "name": "vehicle"}
])

categories_dict = {
	"human"     : 0,
	"bicycle"   : 1,
	"motorcycle": 2,
	"vehicle"   : 3
}


def yolo_to_voc(width, height, x, y, w, h):
	xmax = int((x * width)  + (w * width) / 2.0)
	xmin = int((x * width)  - (w * width) / 2.0)
	ymax = int((y * height) + (h * height) / 2.0)
	ymin = int((y * height) - (h * height) / 2.0)
	return np.array([xmin, ymin, xmax, ymax])


def read_gt(size, gt_path):
	gt_c = []
	gt_b = []
	with open(gt_path, "r") as in_file:
		for line in in_file.read().splitlines():
			items = line.split(" ")
			gt_c.append(categories_dict[items[1]])
			# print(yolo_to_voc(size[0], size[1], float(items[1]), float(items[2]), float(items[3]), float(items[4])))
			gt_b.append(np.array([int(items[2]), int(items[3]), int(items[4]), int(items[5])]))
	
	return gt_c, gt_b


def read_pred(size, pred_path):
	pred_c = []
	pred_b = []
	pred_s = []
	with open(pred_path, "r") as in_file:
		for line in in_file.read().splitlines():
			items = line.split(" ")
			
			# Threshold conf
			if float(items[5]) < 0.4:
				continue
			
			pred_c.append(int(items[0]))
			pred_b.append(yolo_to_voc(size[0], size[1], float(items[1]), float(items[2]), float(items[3]), float(items[4])))
			pred_s.append(float(items[5]))
	
	return pred_c, pred_b, pred_s


def load_images_size(image_size_file):
	list_img_size = {}
	with open(image_size_file, "r") as in_file:
		for line in in_file.read().splitlines():
			items = line.split(" ")
			list_img_size[items[0]] = {"width": int(items[1]), "height": int(items[2])}
	return list_img_size


def evaluation_one(gt_file, pred_file, month):
	with open(gt_file, "rb") as f:
		gth_dict = pickle.load(f)
	
	with open(pred_file, "rb") as f:
		pred_dict = pickle.load(f)
	
	conf_dict  = {}
	batch_size = 64
	a_thres    = 0.3
	c_min      = 0.15
	c_max      = 0.7
	lr         = 0.001
	c          = 0.7
	for month, _ in tqdm(gth_dict.items()):
		
		image_ids_det = []
		image_ids_gt  = []
		gt_boxes      = []
		gt_classes    = []
		det_boxes     = []
		det_classes   = []
		det_scores    = []
		
		for key_img, _ in gth_dict[month].items():
			if key_img in pred_dict[month]:
				if len(pred_dict[month][key_img]["boxes"]) > 0:
					# image_ids_gt.append(key_img)
					# gt_boxes.append(np.array(gth_dict[month][key_img]["boxes"]))
					# gt_classes.append(np.array(gth_dict[month][key_img]["labels"]))
					# pred_dict[month][key_img]["score"] = np.linspace(1.0, 1.0, len(pred_dict[month][key_img]["labels"]))
					det_b = []
					det_c = []
					det_s = []
					# print(det_b, det_c, det_s)
					for i, s in enumerate(pred_dict[month][key_img]["scores"]):
						if s >= c:
							det_b.append(pred_dict[month][key_img]["boxes"][i])
							det_c.append(pred_dict[month][key_img]["labels"][i])
							det_s.append(pred_dict[month][key_img]["scores"][i])
					
					if len(det_b) == 0:
						continue
					
					image_ids_gt.append(key_img)
					gt_boxes.append(np.array(gth_dict[month][key_img]["boxes"]))
					gt_classes.append(np.array(gth_dict[month][key_img]["labels"]))
					
					image_ids_det.append(key_img)
					det_boxes.append(np.array(det_b))
					det_classes.append(np.array(det_c))
					det_scores.append(np.array(det_s))
					
			if len(image_ids_gt) >= batch_size:
				# Convert all lists to numpy arrays
				image_ids_gt  = np.array(image_ids_gt,  dtype=object)
				gt_boxes      = np.array(gt_boxes,      dtype=object)
				gt_classes    = np.array(gt_classes,    dtype=object)
				image_ids_det = np.array(image_ids_det, dtype=object)
				det_boxes     = np.array(det_boxes,     dtype=object)
				det_classes   = np.array(det_classes,   dtype=object)
				det_scores    = np.array(det_scores,    dtype=object)
				# print(det_boxes, det_classes, det_scores)
				
				# Convert ground truth list to dict
				groundtruth_dict = coco_tools.ExportGroundtruthToCOCO(
					image_ids_gt, gt_boxes, gt_classes, categories
				)
				
				# Convert detections list to dict
				detections_list = coco_tools.ExportDetectionsToCOCO(
					image_ids_det, det_boxes, det_scores, det_classes, categories
				)
				
				groundtruth = coco_tools.COCOWrapper(groundtruth_dict)
				detections  = groundtruth.LoadAnnotations(detections_list)
				evaluator   = coco_tools.COCOEvalWrapper(groundtruth, detections)
				metrics, per_category_ap = evaluator.ComputeMetrics(include_metrics_per_category=False)
				
				a_curr  = metrics["Precision/mAP"]
				delta_a = a_thres - a_curr
				
				if a_curr > a_thres:
					c  = min(c_max, c - lr * delta_a)
					# lr = lr * 2
				else:
					c  = max(c_min, c - lr * delta_a)
					# lr = lr / 2
				print(f"{month} = {c}")
				
				image_ids_det = []
				image_ids_gt  = []
				gt_boxes      = []
				gt_classes    = []
				det_boxes     = []
				det_classes   = []
				det_scores    = []
		
		conf_dict = c
		
	return conf_dict


def run_evaluation(opt):
	header = [
		"Name",
		"Precision/mAP",
		"Precision/mAP@.50IOU",
		"Precision/mAP@.75IOU",
		"Precision/mAP (small)",
	    "Precision/mAP (medium)",
		"Precision/mAP (large)",
		"Recall/AR@1",
		"Recall/AR@10",
		"Recall/AR@100",
		"Recall/AR@100 (small)",
	    "Recall/AR@100 (medium)",
		"Recall/AR@100 (large)"
	]
	
	# months = ["jan", "mar", "apr", "may", "jun", "jul", "aug", "sep"]
	# for month in months:
	# 	print(month)
	# 	metrics = evaluation_one(gt_file, pred_file, month)
	# 	print(metrics)
	metrics = evaluation_one(opt.gt_file, opt.pred_file, "")
	print(metrics)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--gt-file",     default=os.path.join("inference", "groundtruth_val.pkl"), type=str, help="Ground-truth")
	parser.add_argument("--output-path", default=os.path.join("inference", "output"),              type=str, help="Output path")
	parser.add_argument("--pred-file",   default=os.path.join("inference", "predictions.pkl"),     type=str, help="Prediction")
	opt = parser.parse_args()
	run_evaluation(opt)
