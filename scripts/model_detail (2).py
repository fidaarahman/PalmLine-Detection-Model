import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="path to your model")
interpreter.allocate_tensors()
output_details = interpreter.get_output_details()

print("Output shape:", output_details[0]['shape'])  # ← should be [1, 9, 8400]

        