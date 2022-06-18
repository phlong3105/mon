---
layout      : default
title       : AIC22 Track 4
parent	    : AI City Challenge
grand_parent: Challenges
has_children: false
has_toc     : false
permalink   : /challenges/aic/aic22_track4
---

![data/aic22_track4.gif](data/aic22_track4.gif)

# AIC22 Track 4: Multi-Class Product Counting & Recognition for Automated Retail Checkout

[Website](https://www.aicitychallenge.org/2022-challenge-tracks){: .btn .fs-3 .mb-4 .mb-md-0 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

---

A growing application of AI and computer vision is in the retail industry. Of
the various problems that can be addressed, this track focuses on accurate and
automatic check-out in a retail store. The challenge stems from the real-world
scenario of occlusion, movement, similarity in items being scanned, novel SKUs
that are created seasonally, and the cost of misdetection and misclassification.

Participating teams will count and identify products as they move along a Retail
checkout conveyor belt. For example, given a conveyor belt snapshot/video teams
will count and identify all products. Products may be occluded or very similar
to each other. The training set will be composed of both real-world data and
synthetic data. The usage of synthetic data is encouraged as it can be simulated
under various environments and can produce large training data sets. Evaluation
will be on a test set of objects not included in training for both for the
closed and open-world scenarios. To maximize the practical value of the outcome
from this track, both product recognition effectiveness and the program
execution efficiency will contribute to the final score for each participating
team. The team with the highest combined efficiency and effectiveness score will
be declared the winner of this track.

## Important Dates

- **Data sets shared with participants**: 02/27/2022


- **Evaluation server open to submissions**: 03/15/2022


- **Challenge track submissions due**: 04/09/2022 (11:59 PM, Pacific Time)
	- _Evaluation submission is closed and rankings are finalized._


- **Workshop papers due**: 04/13/2022 (11:59 PM, Pacific Time)
	- _Since our review is not double-blind, papers should be submitted in
	  final/camera-ready form._


- **Final papers due**: 04/18/2022 (11:59 PM, Pacific Time)
	- _All camera ready paper should be uploaded to CMT to be published by
	  CVPR 2022. The accepted workshop papers will be accessible online at IEEE
	  Xplore Digital Library and CVF Open Access._


- **Open source on GitHub (training code + testing code + additional
  annotation)**: 04/28/2022 (11:59 PM, Pacific Time)
	- _All the competitors/candidates for awards MUST release their code for
	  validation before decision of awardees. The performance on the leaderboard
	  has to be reproducible without the use of external data._

## Data 

In test scenario, **the camera is mounted above the checkout counter and facing
straight down while a customer is pretending to perform a checkout action by
“scanning” objects in front of the counter in a natural manner**. Several
different customers participated and each of them scanned slightly differently
to add to the complexity. **There is a shopping tray placed under the camera to
indicate where the AI model should focus. Participating customers might or might
not place objects on the tray**. One video clip contains several complete
scanning actions, involving one or more items. In summary, the dataset contains:

- Training set – 116,500 synthetic images with classification and segmentation
  labels.
- Test set A – 20% of recorded test video
- Test set B – 80% of recorded test video

**Synthetic data** is provided for model training. There are 116,500 synthetic
images from over 100 3D objects. We use synthetic data because they can form
large-scale training sets under various environments. Following the generation
pipeline in [5], images are filmed with random attributes, i.e., random object
orientation, camera pose, and lighting. Random background images, which are
selected from Microsoft COCO [6], are used to increase the dataset diversity.
The labeling format for synthetic data is “.jpg”, e.g., for the file
00001_697.jpg:

- 00001 means the object has class ID 1, and
- 697 is a counter, i.e., this is the 697th image.

We also provide segmentation labels for these images. For example,
“00001_697_seg.jpg” is the segmentation label for image “00001_697.jpg”. The
white area denotes the object area while the black shows the background.

## Tasks

Teams will be provided with the training set (with labels) and test set A (
without labels). Test set B will be reserved for later testing.

Participating teams need to train a model using the training set provided and
classify the merchandise item held by the customer in each of the video clips.
Teams can use test set A to develop inference code. Teams then submit results
for test set A to our online evaluation server to be shown on the public leader
board for performance tracking. The public leader board only provides a way for
a team to evaluate and improve their systems and the ranking will NOT determine
the winners of this track.

Test set B is reserved for later testing. Top performers on the public ranking
board will be invited to submit functional training and inference code.
Organizers will test the submitted code against dataset B and the final winner
will be determined on the model’s performance against Test set B. If there is a
tie between top teams, efficiency of inference code will be used as the 
tiebreaker, where the team with the most efficient model will be the winner. 
Teams wishing to be considered for evaluation on dataset B must also make their
training and inference codes publicly available.

**Teams that wish to be listed in the public leader board and win the challenge
awards are NOT allowed to use any external data for either training or
validation**. The winning teams and runners-up are required to submit their
training and testing codes for verification after the challenge submission
deadline in order to ensure that no external data was used for training and the
tasks were performed by algorithms and not humans.

## Submission

To be ranked on the public leader board of test set A, one text file should be
submitted to the online evaluation system containing, on each line, details of
one identified activity, in the following format (values are space-delimited):

```text
<video_id> <class_id> <timestamp>
```

Where:

- `<video_id>` is the video numeric identifier, starting with 1. It represents
  the position of the video in the list of all track 4 test set A videos, sorted
  in alphanumeric order.
- `<class_id>` is the object numeric identifier, starting with 1.
- `<timestamp>` is the time in the video when the object was first identified,
  in seconds. The timestamp is an integer and represents a time when the item is
  within the region of interest, i.e., over the white tray. Each object should
  only be identified once while it passes through the region of interest.

## Evaluation

Evaluation for track 4 will be based on model identification performance,
measured by the **F1-score**. For the purpose of computing the F1-score, a
true-positive (TP) identification will be considered when an object was
correctly identified within the region of interest, i.e., the object class was
correctly determined, and the object was identified within the time that the
object was over the white tray. A false-positive (FP) is an identified object
that is not a TP identification. Finally, a false-negative (FN) identification
is a ground-truth object that was not correctly identified.

## Methods

| Status | Method | Team ID | Rank | Architecture | Date | Publication |
|:------:|--------|:-------:|:----:|--------------|------|-------------|
|        |        |         |      |              |      |             |


| ![data/aic22_track4_round_01.png](data/aic22_track4_round_01.png) |
|:-----------------------------------------------------------------:|
|                   Leaderboard of the 1st round.                   |
