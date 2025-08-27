import onnx
import tensorflow as tf
from onnx_tf.backend import prepare

# Load ONNX model
onnx_model_path = "path of your onnx file"
onnx_model = onnx.load(onnx_model_path)

# Convert ONNX to TensorFlow
tf_rep = prepare(onnx_model)  # Convert ONNX to TensorFlow
saved_model_path = "path where u want to save the model"
tf_rep.export_graph(saved_model_path)  # Save as TensorFlow model

print(f"TensorFlow model saved at: {saved_model_path}")

# Convert TensorFlow to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Optional: Enable quantization
tflite_model = converter.convert()

# Save the TFLite model
tflite_model_path = "path where u want to save the model"
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print(f"TFLite model saved at: {tflite_model_path}")
