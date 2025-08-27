import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="C:/Users/fidau/Downloads/Palm lines recognition.v10i.yolov8/best_saved_model/best_float32.tflite")
interpreter.allocate_tensors()
output_details = interpreter.get_output_details()

print("Output shape:", output_details[0]['shape'])  

        