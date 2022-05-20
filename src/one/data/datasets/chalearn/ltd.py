#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The LTD dataset used in the Seasons in Drift Challenge at ECCV'22 is an
extension of an existing concept drift dataset and spans 188 days in the period
of 14th May 2020 to 30th of April 2021, with a total of 1689 2-minute clips
sampled at 1fps with associated bounding box annotations for 4 classes
(Human, Bicycle, Motorcycle, Vehicle). The collection of this dataset has
included data from all hours of the day in a wide array of weather conditions
overlooking the harbor-front of Aalborg, Denmark. In this dataset depicts the
drastic changes of appearance of the objects of interest as well as the scene
over time in a static surveillance context to develop robust algorithms for
real-world deployment.

Statistics:
	######### Object Size Grouping Scheme #########
	Small (<1024 pixels)
	Medium (1025-9695 pixels)
	Large (>9696 pixels)
	
	############### Subset Overview ###############
	Subset name   :  Full-All
	Clips         :  1689
	Different days:  188
	Timespan      :  2020-05-14 - 2021-04-30
	-------------- Object Presence ----------------
	Empty frames  : 844638 (78.9937217499792 %)
	Frames /w obj : 224609 (21.00627825002081 %)
	Total frames  : 1069247
	
	########### Object Distributions ##############
	All*          : 6868067
	bicycle       : 293280
	human         : 5841139
	motorcycle    : 32393
	vehicle       : 701255
	Unique Objects: 143294
	------------------- Small ---------------------
	All*          : 6092590
	bicycle       : 288081
	human         : 5663804
	motorcycle    : 27153
	vehicle       : 113552
	------------------- Medium --------------------
	All*          : 37468
	bicycle       : 7
	human         : 454
	vehicle       : 37007
	------------------- Large ---------------------
	All*          : 738009
	bicycle       : 5192
	human         : 176881
	motorcycle    : 5240
	vehicle       : 550696
	###############################################

References:
	https://chalearnlap.cvc.uab.cat/dataset/43/description/
"""

from __future__ import annotations

__all__ = [

]


# MARK: - Module
