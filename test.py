from flask import Flask
from flask import jsonify
from yolo import YOLO
import os
from PIL import Image

# create instance of the flask class
app = Flask(__name__)
app.config["DEBUG"] = True

# initialize instance of class YOLO in memory
yolo_m1 = YOLO()
# yolo_m2 = YOLO()

# set model paths
yolo_m1.model_path ='model_data/final_model.h5'
# yolo_m2.model_path ='model_data/model_2/with_6_categories.h5'

# set anchors and txt_paths
yolo_m1.anchors_path = 'model_data/new_anchors.txt'
yolo_m1.classes_path = 'model_data/new_classes.txt'
yolo_m1.class_names = yolo_m1._get_class()
# yolo_m2.anchors_path = 'model_data/model_2/txts/anchors.txt'
# yolo_m2.classes_path = 'model_data/model_2/txts/classes.txt'
yolo_m1.anchors = yolo_m1._get_anchors()
# load models in memory
yolo_m1.boxes, yolo_m1.scores, yolo_m1.classes = yolo_m1.generate()
print("model 1 loaded successfully...")

# yolo_m2.boxes, yolo_m2.scores, yolo_m2.classes = yolo_m2.generate()
# print("model 2 loaded successfully...")


def detection_function():
	predictions = {}
	directory = yolo_m1.test_images_directory
	files = os.listdir(directory)
	for image_name in files:
		try:
			image_for_model1 = Image.open(directory+image_name)
# 			image_for_model2 = Image.open(directory+image_name)
		except:
			print('Open Error!',image_name,'Try again!')
			continue
		else:
			print('Running model 1 on image :',image_name)
			yolo_m1.detect_image(image_for_model1, image_name)
			
# 			if image_name not in predictions:
# 				predictions[image_name] = []
# 			predictions[image_name].append(pred_1)

# 			print('Running model 2 on image :',image_name)
# 			pred_2 = yolo_m2.detect_image(image_for_model2, image_name)
			
# 			predictions[image_name].append(pred_2)
# 	print('predictions', predictions)
	return "success"

@app.route('/scan_images', methods=['POST'])
def call_model_1():
	return detection_function()


@app.route('/delete_images', methods=['POST'])
def call_model_2():
	os.system("rm test_images/*")
	os.system("rm predictions/*")
	return "done"

if __name__ == '__main__':
	app.run(use_reloader=False, host='0.0.0.0')