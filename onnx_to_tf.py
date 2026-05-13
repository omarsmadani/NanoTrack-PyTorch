import os
import numpy as np
import onnx2tf

ONNX_DIR = "./models/onnx"
TF_DIR = "./models/tf"

os.makedirs(TF_DIR, exist_ok=True)

models = [
    ("nanotrack_backbone_sim.onnx", "nanotrack_backbone_tf"),
    ("nanotrack_head_sim.onnx",     "nanotrack_head_tf"),
]

for onnx_name, tf_name in models:
    onnx_path = os.path.join(ONNX_DIR, onnx_name)
    tf_path   = os.path.join(TF_DIR, tf_name)
    print(f"\n--- Converting {onnx_name} -> {tf_path} ---")
    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_path,
        not_use_onnxsim=True,
        check_onnx_tf_outputs_elementwise_close=False,
        non_verbose=True,
    )
    print(f"Saved: {tf_path}")

print("\nAll conversions complete.")
