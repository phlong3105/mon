import cv2


def compare_images(image_path_1, image_path_2):
	# Read the two images
	img1 = cv2.imread(image_path_1)
	img2 = cv2.imread(image_path_2)
	
	if img1 is None or img2 is None:
		print("Error: One of the image paths is incorrect.")
		return
	
	# Ensure the images have the same size
	if img1.shape != img2.shape:
		print("Images have different sizes, resizing the second image to match the first.")
		img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
	
	# Calculate the absolute difference between the images
	diff = cv2.absdiff(img1, img2)
	
	# Convert the difference to grayscale to better visualize the differences
	diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
	
	# Apply a threshold to get a binary difference image
	_, diff_threshold = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)
	
	# Find contours of the differences
	contours, _ = cv2.findContours(diff_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# Draw the contours on the original images to highlight the differences
	img1_copy = img1.copy()
	img2_copy = img2.copy()
	cv2.drawContours(img1_copy, contours, -1, (0, 0, 255), 2)  # Red contours on img1
	cv2.drawContours(img2_copy, contours, -1, (0, 0, 255), 2)  # Red contours on img2
	
	# Display the images and differences
	cv2.imshow('Image 1 with Differences', img1_copy)
	cv2.imshow('Image 2 with Differences', img2_copy)
	cv2.imshow('Absolute Difference', diff)
	cv2.imshow('Thresholded Differences', diff_threshold)
	
	# Wait for a key press and close all windows
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	# Provide paths to the images to compare
	image_path_1 = 'path_to_first_image.jpg'
	image_path_2 = 'path_to_second_image.jpg'
	
	compare_images(image_path_1, image_path_2)
