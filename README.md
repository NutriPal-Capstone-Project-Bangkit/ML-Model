# Object Detection Model
## Overview
This object detection model is designed to detect the position of nutrition labels on product packaging. It leverages transfer learning using TensorFlow's Object Detection API with the base model SSD MobileNet V2 320x320. The primary purpose of this model is to create bounding boxes around nutrition labels, facilitating the OCR process.
## Usage
Steps to generate the model:
1. Install all the dependency
    - Tensorflow’s Object detection API 
    - Roboflow
    - tf-models-official==2.15.0
    - protobuf==3.20.0
2. Load the dataset from roboflow
    ```sh
    from roboflow import Roboflow
    ```
3. Preprocess and prepare dataset for training
4. Download and Prepare Pretrained Model Checkpoint -  SSD MobileNet V2 FPNLite (320x320), 
    ```sh
    !wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
    !tar -xf ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz
    !if [ -d "models/research/object_detection/test_data/checkpoint" ]; then rm -Rf models/research/object_detection/test_data/checkpoint; fi
    !mkdir models/research/object_detection/test_data/checkpoint
    !mv ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint models/research/object_detection/test_data/
    ```
5. Fine tuned the model with these layers
    ```sh
    prefixes_to_train = [
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead',
        'WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead']
    ```
6. Test model and see the result
7. Save and convert to TFLite
    ```sh
    converter = tf.lite.TFLiteConverter.from_saved_model('tflite/saved_model')
    ```
## Dataset Overview
- Total class : 1 (nutrition-label)
- Total photos : 244
- Train set : 70 % (170 images)
- Test set : 30% (74 images)

## Result
- Loss visualization
  ![image alt](https://github.com/NutriPal-Capstone-Project-Bangkit/ML-Model/blob/main/object_detection_model/loss.png)
  
- Detection result

  ![image alt](https://github.com/NutriPal-Capstone-Project-Bangkit/ML-Model/blob/main/object_detection_model/result1.png)
  ![image alt](https://github.com/NutriPal-Capstone-Project-Bangkit/ML-Model/blob/main/object_detection_model/result2.png)


# Generative AI
## Overview
This model provides personalized recommendations for daily nutrition needs, consumption limits for certain products, and advice on how to achieve user-specific goals based on exercise activity levels, height, weight, age, gender, and more.
## Use cases
The model generates personalized nutrition advice, such as:
- Daily calorie and macronutrient recommendations (fats, carbs, proteins)
- Advice on adjusting total daily energy expenditure (TDEE) based on goals like weight loss, weight gain, or weight maintenance
- Suggested food and product intake based on user activity level, age, height, weight, and specific goals

## Dataset Overview (JSONL Format)
Input Features:
- Exercise activity level (e.g., casual, moderately active, active, regular)
- Height
- Weight
- Age
- Gender
- Specific goals (e.g., weight loss, weight gain, or maintaining weight)

Output Features:
- BMR (Basal Metabolic Rate) calculation
- TDEE (Total Daily Energy Expenditure) calculation
- Macronutrient breakdown (fats, carbs, protein) based on goals
- Recommendations

## Usage
Steps to Fine-Tune the Model in Vertex AI:
1. Prepare Dataset (JSONL Format)
2. Upload dataset to google cloud storage
3. Activate vertex AI in Google Cloud Console
4. Fine tune by choose the Gemini 1.5 Pro model as base model in Generative AI Studio
5.  Deploy the Fine-Tuned Model and Get the Endpoint API




















