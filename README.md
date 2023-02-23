# EAI_self_driving
[Github link](https://github.com/alen-smajic/Real-time-Object-Detection-for-Autonomous-Driving-using-Deep-Learning)

## Data
[Here](https://bdd-data.berkeley.edu/)

## Model Weight
Download from [Here](https://drive.google.com/drive/folders/1NGOnVfMcpzedTR0NurP05FXd8zxsF9JI?usp=sharing)

Place in those position:

    Faster R-CNN\training\ckpt-26.data-00000-of-00001
    Faster R-CNN\models\inference_graph\checkpoint\ckpt-0.data-00000-of-00001
    Faster R-CNN\models\inference_graph\saved_model\variables\variables.data-00000-of-00001

## Prediction  

Need to add Tensorflow 2 object detection API. Look at this [link](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb) or [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md)

Images

    python3 detect_objects.py --model_path models/inference_graph/saved_model --path_to_labelmap models/label_map.pbtxt --images_dir data/samples/images/ --save_output  
    
Video

    python3 detect_objects.py --model_path models/inference_graph/saved_model --path_to_labelmap models/label_map.pbtxt --video_input --video_path data/video_samples/1.mov --save_output

Parameters:
* ```--model_path``` System path to the frozen detection model, default=```models/efficientdet_d0_coco17_tpu-32/saved_model```
* ```--path_to_labelmap``` Path to the labelmap (.pbtxt) file
* ```--class_ids``` IDs of classes to detect, expects string with IDs separated by ","
* ```--threshold``` Detection Ttreshold, default=```0.4```
* ```--images_dir``` Directory to input images, default=```'data/samples/images/```
* ```--video_path``` Path to input video
* ```--output_directory``` Path to output images/video, default=```data/samples/output```
* ```--video_input``` Flag for video input, default=```False```
* ```--save_output``` Flag for saveing images and video with visualized detections, default=```False```

## Problem
1. Pototype layer haven't correctly imlement
2. Model only save `faster_rcnn_fe_extractor`
3. Currently omly try on image
4. When try on prediction, `detect_objects.py` doesn't have any output.
