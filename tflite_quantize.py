import os
import cv2
import numpy as np
import tensorflow as tf

BACKBONE_SAVED_MODEL = "./models/tf/nanotrack_backbone_tf"
HEAD_SAVED_MODEL     = "./models/tf/nanotrack_head_tf"
OUT_DIR              = "./models/coral"
NUM_CALIB_SAMPLES    = 200

os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Build calibration crops from the available videos.
# Mimics the tracker's subwindow extraction: random 255x255 BGR crops,
# cast to float32, no normalization (model expects raw [0-255] values).
# ---------------------------------------------------------------------------
def collect_calibration_frames(n=NUM_CALIB_SAMPLES):
    videos = [
        "./bin/Video.mp4",
        "./bin/Video2.mp4",
        "./bin/Video3.mp4",
    ]
    frames = []
    per_video = max(1, n // len(videos))
    for path in videos:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            continue
        indices = np.linspace(0, total - 1, per_video, dtype=int)
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            h, w = frame.shape[:2]
            # Random 255x255 crop with padding if needed
            pad = 255
            frame_padded = cv2.copyMakeBorder(frame, pad, pad, pad, pad,
                                              cv2.BORDER_REFLECT)
            cy = np.random.randint(pad, pad + h)
            cx = np.random.randint(pad, pad + w)
            crop = frame_padded[cy-127:cy+128, cx-127:cx+128]  # 255x255
            crop = cv2.resize(crop, (255, 255))
            frames.append(crop.astype(np.float32))
        cap.release()
        if len(frames) >= n:
            break
    frames = frames[:n]
    print(f"Collected {len(frames)} calibration frames")
    return frames

calibration_frames = collect_calibration_frames()

# ---------------------------------------------------------------------------
# Backbone: input [1, 255, 255, 3] NHWC float32
# ---------------------------------------------------------------------------
def backbone_rep_dataset():
    for frame in calibration_frames:
        inp = frame[np.newaxis, ...]          # [1, 255, 255, 3]
        yield [inp]

print("\n--- Quantizing backbone ---")
converter = tf.lite.TFLiteConverter.from_saved_model(BACKBONE_SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = backbone_rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.int8

backbone_tflite = converter.convert()
backbone_path = os.path.join(OUT_DIR, "nanotrack_backbone_int8.tflite")
with open(backbone_path, "wb") as f:
    f.write(backbone_tflite)
print(f"Saved: {backbone_path} ({os.path.getsize(backbone_path)//1024} KB)")

# Verify quantization
interp = tf.lite.Interpreter(model_content=backbone_tflite)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f"  input:  {inp['name']} {inp['shape']} {inp['dtype'].__name__}  scale={inp['quantization'][0]:.6f} zp={inp['quantization'][1]}")
print(f"  output: {out['name']} {out['shape']} {out['dtype'].__name__}  scale={out['quantization'][0]:.6f} zp={out['quantization'][1]}")

# ---------------------------------------------------------------------------
# Head calibration: run quantized backbone to get real int8 features,
# then dequantize for the head's float32 SavedModel converter.
# ---------------------------------------------------------------------------
print("\n--- Generating head calibration data via backbone ---")

# Use the float32 backbone SavedModel to get real feature distributions
bb_float = tf.lite.Interpreter(
    model_path=os.path.join(BACKBONE_SAVED_MODEL,
                            "nanotrack_backbone_sim_float32.tflite")
    if os.path.exists(os.path.join(BACKBONE_SAVED_MODEL,
                                   "nanotrack_backbone_sim_float32.tflite"))
    else None
)

# Fall back to using the just-quantized backbone if float32 tflite not present
# Actually just load the float32 tflite we already have
bb_float32_path = "./models/tf/nanotrack_backbone_tf/nanotrack_backbone_sim_float32.tflite"
bb_float = tf.lite.Interpreter(model_path=bb_float32_path)
bb_float.allocate_tensors()
bb_in_idx  = bb_float.get_input_details()[0]['index']
bb_out_idx = bb_float.get_output_details()[0]['index']

zf_samples = []
xf_samples = []

for frame in calibration_frames:
    # Search features (full 255x255 crop)
    x_nhwc = frame[np.newaxis, ...]
    bb_float.set_tensor(bb_in_idx, x_nhwc)
    bb_float.invoke()
    xf = bb_float.get_tensor(bb_out_idx).copy()   # [1, 48, 16, 16]
    xf_samples.append(xf.transpose(0, 2, 3, 1))   # -> [1, 16, 16, 48]

    # Template features (centre 127x127 of the crop, resized to 255x255)
    h, w = frame.shape[:2]
    z_crop = frame[64:191, 64:191]                 # 127x127 centre
    z_255  = cv2.resize(z_crop, (255, 255)).astype(np.float32)
    z_nhwc = z_255[np.newaxis, ...]
    bb_float.set_tensor(bb_in_idx, z_nhwc)
    bb_float.invoke()
    zf_full = bb_float.get_tensor(bb_out_idx).copy()  # [1, 48, 16, 16]
    # Take centre 8x8 spatial (equivalent to template crop)
    zf_center = zf_full[:, :, 4:12, 4:12]             # [1, 48, 8, 8]
    zf_samples.append(zf_center.transpose(0, 2, 3, 1)) # -> [1, 8, 8, 48]

print(f"Generated {len(xf_samples)} xf and {len(zf_samples)} zf calibration tensors")

# ---------------------------------------------------------------------------
# Head: inputs zf [1,8,8,48] and xf [1,16,16,48], outputs cls/loc
# ---------------------------------------------------------------------------
def head_rep_dataset():
    n = min(len(zf_samples), len(xf_samples))
    for i in range(n):
        yield [zf_samples[i], xf_samples[i]]

print("\n--- Quantizing head ---")
converter = tf.lite.TFLiteConverter.from_saved_model(HEAD_SAVED_MODEL)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = head_rep_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type  = tf.int8
converter.inference_output_type = tf.int8

head_tflite = converter.convert()
head_path = os.path.join(OUT_DIR, "nanotrack_head_int8.tflite")
with open(head_path, "wb") as f:
    f.write(head_tflite)
print(f"Saved: {head_path} ({os.path.getsize(head_path)//1024} KB)")

interp = tf.lite.Interpreter(model_content=head_tflite)
interp.allocate_tensors()
for d in interp.get_input_details():
    print(f"  input:  {d['name']} {d['shape']} {d['dtype'].__name__}  scale={d['quantization'][0]:.6f} zp={d['quantization'][1]}")
for d in interp.get_output_details():
    print(f"  output: {d['name']} {d['shape']} {d['dtype'].__name__}  scale={d['quantization'][0]:.6f} zp={d['quantization'][1]}")

print("\nDone. INT8 models ready for edgetpu_compiler.")
