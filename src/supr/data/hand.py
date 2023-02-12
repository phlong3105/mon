#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements hand detection using :mod:`mediapipe` package."""

from __future__ import annotations

__all__ = [
	"Hands", "HandsEstimator",
]

import cv2
import mediapipe as mp
import numpy as np


# region Hands

class Hands:
	"""The hands object.
	
	Args:
		multi_hand_landmarks: Hand landmarks on each detected hand.
        multi_hand_landmarks_norm: Normalized Hand landmarks on each detected
            hand.
        multi_hand_world_landmarks: Hand landmarks on each detected hand in
            real-world 3D coordinates that are in meters with the origin at the
            hand's approximate geometric center.
        multi_handedness: Handedness (left v.s. right hand) of the detected
            hand.
        frame_index: Index of frame when the Detection is created. Defaults to
            None.
	"""
	
	def __init__(
		self,
		multi_hand_landmarks,
		multi_hand_landmarks_norm,
		multi_hand_world_landmarks,
		multi_handedness,
		frame_index: int | None = None,
	):
		super().__init__()
		self.multi_hand_landmarks       = multi_hand_landmarks
		self.multi_hand_landmarks_norm  = multi_hand_landmarks_norm
		self.multi_hand_world_landmarks = multi_hand_world_landmarks
		self.multi_handedness           = multi_handedness
		self.frame_index                = frame_index
		
	@property
	def multi_finger_landmarks(self) -> list | None:
		if self.multi_hand_landmarks:
			fingers_lms = []
			for hand_landmarks in self.multi_hand_landmarks:
				fingers_lms.append(hand_landmarks[5:])
			return fingers_lms
		else:
			return None
		
	def draw(self, drawing: np.ndarray) -> np.ndarray:
		for hand_landmarks in self.multi_hand_landmarks:
			[cv2.circle(drawing, l, 10, (139, 0, 0), cv2.FILLED) for l in hand_landmarks]
		for hand_landmarks in self.multi_hand_landmarks_norm:
			mp.solutions.drawing_utils.draw_landmarks(
				image         = drawing,
				landmark_list = hand_landmarks,
				connections   = mp.solutions.hands.HAND_CONNECTIONS
			)
		return drawing
	
# endregion


# region HandsEstimator

class HandsEstimator:
	"""MediaPipe Hand Estimator.
	
	MediaPipe Hands processes an RGB image and returns the hand landmarks and
	handedness (left v.s. right hand) of each detected hand.
	
	Note that it determines handedness assuming the input image is mirrored,
	i.e., taken with a front-facing/selfie camera
	(https://en.wikipedia.org/wiki/Front-facing_camera) with images flipped
	horizontally. If that is not the case, use, for measurement,
	cv2.flip(image, 1)
	to flip the image first for a correct handedness output.
	 
	Please refer to https://solutions.mediapipe.dev/hands#python-solution-api
	for usage examples.
	
	Args:
		static_image_mode: Whether to treat the input images as a batch of
			static and possibly unrelated images, or a video stream. See details
			in https://solutions.mediapipe.dev/hands#static_image_mode.
			Defaults to False.
        max_num_hands: Maximum number of hands to detect. See details in
            https://solutions.mediapipe.dev/hands#max_num_hands.
            Defaults to 2.
        model_complexity: Complexity of the hand landmark model: 0 or 1.
            Landmark accuracy as well as inference latency generally goes up
            with the model complexity. See details in
            https://solutions.mediapipe.dev/hands#model_complexity.
	        Defaults to 1.
        min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand
            measurement to be considered successful. See details in
            https://solutions.mediapipe.dev/hands#min_detection_confidence.
            Defaults to 0.5.
        min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
            hand landmarks to be considered tracked successfully. See details in
            https://solutions.mediapipe.dev/hands#min_tracking_confidence.
            Defaults to 0.5.
        hand_tracker: MediaPipe Hand object.
	"""
	
	def __init__(
		self,
		static_image_mode       : bool  = False,
		max_num_hands           : int   = 2,
		model_complexity        : int   = 1,
		min_detection_confidence: float = 0.5,
		min_tracking_confidence : float = 0.5,
		*args, **kwargs
	):
		super().__init__()
		self.static_image_mode          = static_image_mode
		self.max_num_hands              = max_num_hands
		self.model_complexity           = model_complexity
		self.min_detection_confidence   = min_detection_confidence
		self.min_tracking_confidence    = min_tracking_confidence
		
		self.hand_tracker = mp.solutions.hands.Hands(
			static_image_mode        = self.static_image_mode,
			max_num_hands            = self.max_num_hands,
			model_complexity         = self.model_complexity,
			min_detection_confidence = self.min_detection_confidence,
			min_tracking_confidence  = self.min_detection_confidence
		)
		
	def estimate(self, indexes: np.ndarray, images: np.ndarray) -> list[Hands]:
		"""Detect hands landmarks in images.

        Args:
            indexes: Image indexes.
            images: Images of shape [B, H, W, C].

        Returns:
            A list of :class:`Hands` objects.
        """
		return [self.process(idx, img) for idx, img in zip(indexes, images)]
	
	def process(self, index: int, image: np.ndarray) -> Hands | None:
		"""Process an RGB image and returns the hand landmarks and handedness
		of each detected hand.
		
		Args:
			image: An RGB image of shape [H, W, C].
	
	    Raises:
	    	RuntimeError: If the underlying graph throws any error.
	    	ValueError: If the input image isn't three channel RGB.
	
	    Returns:
	    	A `Hands` object that contains:
	        1) A "multi_hand_landmarks_norm" field that contains the hand
	           landmarks on each detected hand.
	        2) A "multi_hand_world_landmarks" field that contains the hand
	           landmarks on each detected hand in real-world 3D coordinates that
	           are in meters with the origin at the hand's approximate geometric
	           center.
	        3) A "multi_handedness" field that contains the handedness
	           (left v.s. right hand) of the detected hand.
		"""
		h, w, c = image.shape
		results = self.hand_tracker.process(image)
		
		if results.multi_hand_landmarks:
			multi_hand_landmarks = []
			for hand_landmarks_norm in results.multi_hand_landmarks:
				hand_landmarks = []
				for id, lm in enumerate(hand_landmarks_norm.landmark):
					cx = int(lm.x * w)
					cy = int(lm.y * h)
					hand_landmarks.append((cx, cy))
				multi_hand_landmarks.append(hand_landmarks)
		
			return Hands(
				multi_hand_landmarks       = multi_hand_landmarks,
				multi_hand_landmarks_norm  = results.multi_hand_landmarks,
				multi_hand_world_landmarks = results.multi_hand_world_landmarks,
				multi_handedness           = results.multi_handedness,
				frame_index                = index,
			)
		else:
			return None

# endregion
