#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This free Teledyne FLIR ADAS Dataset provides fully annotated thermal and
visible spectrum frames for the development of object detection systems using
convolutional neural networks (CNNs). This data was constructed to encourage
research on visible + thermal sensor fusion algorithms ("RGBT") and to empower
the automotive community to create safer and more efficient ADAS and driverless
vehicle systems.

The ability to sense thermal infrared radiation, or heat, provides both
complementary and distinct advantages to existing sensor technologies such as
visible cameras, Lidar, and radar systems. The Teledyne FLIR thermal sensors
can detect and classify in challenging conditions including total darkness, most
fog, smoke, inclement weather, and glare. When combined with visible light data
and distance scanning data from Lidar and radar, thermal data paired with
machine learning creates a more comprehensive detection and classification
system.

- Content: A total of 26,442 fully annotated frames with 520,000 bounding box
           annotations across 15 different object categories.
- Images: 9,711 thermal and 9,233 RGB training/validation images with a
		  suggested training/validation split. Includes 16-bit pre-AGC frames.
- Videos: 7,498 total video frames recorded at 24Hz. 1:1 match between thermal
          and visible frames. Includes 16-bit pre-AGC frames.
- Frame Annotation Label Totals: Over 375,000 annotations in the thermal and
								 visible spectrum.
- Video Annotation Label Totals: Over 145,000 annotations in the thermal and
								visible spectrum
- Label Categories:
	1.  Person
	2.  Bike
	3.  Car
	4.  Motorcycle
	5.  Bus
	6.  Train
	7.  Truck
	8.  Traffic light
	9.  Fire Hydrant
	10. Street Sign
	11. Dog
	12. Skateboard
	13. Stroller
	14. Scooter
	15. Other Vehicle
- Thermal Camera Specifications: Teledyne FLIR Tau 2 640x512, 13mm f/1.0
							     (HFOV 45°, VFOV 37°)
- Visible Camera Specifications: Teledyne FLIR Blackfly S BFS-U3-51S5C (IMX250)
								 camera and a 52.8° HFOV Edmund Optics lens
- Dataset File Format:
	+ Thermal - 14-bit TIFF (no AGC)
	+ Thermal 8-bit JPEG (AGC applied)
	+ RGB - 8-bit JPEG
	+ MSCOCO formatted annotations (JSON)
	+ Conservator formatted annotations (JSON)
	
References:
	https://www.flir.com/oem/adas/adas-dataset-form/
"""

from __future__ import annotations
