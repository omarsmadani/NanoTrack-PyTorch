"""
NanoTrack inference script for Coral Dev Board Mini (Edge TPU).
No PyTorch dependency — uses only NumPy, OpenCV, and TFLite Runtime.

Usage:
    python3 demo_tflite.py --video /path/to/video.mp4
    python3 demo_tflite.py --video 0          # webcam
    python3 demo_tflite.py --video /path/to/img_dir/

Setup on Coral board:
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
    pip install tflite-runtime pycoral numpy opencv-python-headless yacs
"""

import os
import sys
import argparse
import numpy as np
import cv2
from glob import glob

# ---------------------------------------------------------------------------
# Paths (relative to repo root — run from NanoTrack/)
# The _edgetpu.tflite variants are produced by edgetpu_compiler and run on the
# Edge TPU; the plain _int8.tflite variants run on CPU.
# ---------------------------------------------------------------------------
BACKBONE_TPU = "./models/coral/nanotrack_backbone_int8_edgetpu.tflite"
HEAD_TPU     = "./models/coral/nanotrack_head_int8_edgetpu.tflite"
BACKBONE_CPU = "./models/coral/nanotrack_backbone_int8.tflite"
HEAD_CPU     = "./models/coral/nanotrack_head_int8.tflite"

# Tracking hyper-parameters (from configv3.yaml)
EXEMPLAR_SIZE    = 127
INSTANCE_SIZE    = 255
CONTEXT_AMOUNT   = 0.5
STRIDE           = 16
OUTPUT_SIZE      = 16       # head output grid is 16x16 (matches TFLite cls/loc shape)
PENALTY_K        = 0.138
WINDOW_INFLUENCE = 0.455
LR               = 0.348

# ---------------------------------------------------------------------------
# TFLite interpreter setup
# ---------------------------------------------------------------------------
def make_interpreter(model_path, use_tpu=True):
    """Load a TFLite model, using Edge TPU delegate if available."""
    if use_tpu:
        try:
            import tflite_runtime.interpreter as tflite
            # load_delegate lives in tflite_runtime.interpreter; pycoral re-exports
            # it on newer releases but not on the board's tflite_runtime 2.5.0.
            delegate = tflite.load_delegate('libedgetpu.so.1')
            return tflite.Interpreter(
                model_path=model_path,
                experimental_delegates=[delegate]
            )
        except Exception as e:
            print(f"[warn] Edge TPU delegate unavailable ({e}), falling back to CPU")

    # Fallback: plain TFLite (works on any machine for testing)
    try:
        import tflite_runtime.interpreter as tflite
        return tflite.Interpreter(model_path=model_path)
    except ImportError:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=model_path)


# ---------------------------------------------------------------------------
# Image preprocessing (ported from base_tracker.py — no torch dependency)
# ---------------------------------------------------------------------------
def get_subwindow(im, pos, model_sz, original_sz, avg_chans):
    """Crop a context window and resize to model_sz x model_sz.
    Returns uint8 NHWC array [1, model_sz, model_sz, 3].
    """
    sz = original_sz
    im_h, im_w = im.shape[:2]
    c = (original_sz + 1) / 2
    xmin = int(np.floor(pos[0] - c + 0.5))
    xmax = xmin + sz - 1
    ymin = int(np.floor(pos[1] - c + 0.5))
    ymax = ymin + sz - 1

    left   = max(0, -xmin)
    top    = max(0, -ymin)
    right  = max(0, xmax - im_w + 1)
    bottom = max(0, ymax - im_h + 1)

    xmin += left;  xmax += left
    ymin += top;   ymax += top

    if any([top, bottom, left, right]):
        padded = np.full((im_h + top + bottom, im_w + left + right, 3),
                         avg_chans, dtype=np.uint8)
        padded[top:top+im_h, left:left+im_w] = im
        patch = padded[ymin:ymax+1, xmin:xmax+1]
    else:
        patch = im[ymin:ymax+1, xmin:xmax+1]

    if patch.shape[0] != model_sz or patch.shape[1] != model_sz:
        patch = cv2.resize(patch, (model_sz, model_sz))

    return patch[np.newaxis].astype(np.uint8)   # [1, H, W, 3]


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------
def quantize_uint8_to_int8(x_uint8, scale, zero_point):
    """Convert backbone uint8 output to int8 for the head input."""
    q = x_uint8.astype(np.int32) - zero_point
    return np.clip(q, -128, 127).astype(np.int8)


def dequantize(x_int8, scale, zero_point):
    return (x_int8.astype(np.float32) - zero_point) * scale


# ---------------------------------------------------------------------------
# Post-processing (ported from nano_tracker.py — pure NumPy)
# ---------------------------------------------------------------------------
def generate_points(stride, size):
    ori = -(size // 2) * stride
    xs = ori + stride * np.arange(size)
    x, y = np.meshgrid(xs, xs)
    pts = np.stack([x.flatten(), y.flatten()], axis=1).astype(np.float32)
    return pts


def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def convert_score(cls_nhwc):
    # cls_nhwc: [1, H, W, 2] float32
    cls = cls_nhwc[0].reshape(-1, 2)       # [H*W, 2]
    return softmax(cls, axis=1)[:, 1]      # foreground probability


def convert_bbox(loc_nhwc, points):
    # loc_nhwc: [1, H, W, 4] float32
    loc = loc_nhwc[0].reshape(-1, 4)       # [H*W, 4]
    x1 = points[:, 0] - loc[:, 0]
    y1 = points[:, 1] - loc[:, 1]
    x2 = points[:, 0] + loc[:, 2]
    y2 = points[:, 1] + loc[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w  = x2 - x1
    h  = y2 - y1
    return np.stack([cx, cy, w, h], axis=0)  # [4, H*W]


def bbox_clip(cx, cy, w, h, img_shape):
    H, W = img_shape[:2]
    cx = np.clip(cx, 0, W)
    cy = np.clip(cy, 0, H)
    w  = np.clip(w,  10, W)
    h  = np.clip(h,  10, H)
    return cx, cy, w, h


# ---------------------------------------------------------------------------
# Tracker class
# ---------------------------------------------------------------------------
class NanoTrackerTFLite:
    def __init__(self, backbone_path, head_path, use_tpu=True):
        self.bb_interp = make_interpreter(backbone_path, use_tpu)
        self.bb_interp.allocate_tensors()
        self.hd_interp = make_interpreter(head_path, use_tpu)
        self.hd_interp.allocate_tensors()

        # Backbone I/O details
        self.bb_in  = self.bb_interp.get_input_details()[0]
        self.bb_out = self.bb_interp.get_output_details()[0]

        # Head I/O details (find zf/xf by name)
        hd_ins = {d['name']: d for d in self.hd_interp.get_input_details()}
        self.hd_zf = next(d for k, d in hd_ins.items() if 'zf' in k)
        self.hd_xf = next(d for k, d in hd_ins.items() if 'xf' in k)
        hd_outs = self.hd_interp.get_output_details()
        # cls output is the one with 2 channels, loc has 4
        self.hd_cls = next(d for d in hd_outs if d['shape'][-1] == 2)
        self.hd_loc = next(d for d in hd_outs if d['shape'][-1] == 4)

        # Hanning window
        hanning = np.hanning(OUTPUT_SIZE)
        self.window = np.outer(hanning, hanning).flatten()
        self.points = generate_points(STRIDE, OUTPUT_SIZE)

        self.center_pos = None
        self.size = None
        self.channel_average = None
        self.zf = None

    def _run_backbone(self, crop_uint8):
        """Run backbone on a uint8 NHWC crop, return int8 feature map."""
        self.bb_interp.set_tensor(self.bb_in['index'], crop_uint8)
        self.bb_interp.invoke()
        return self.bb_interp.get_tensor(self.bb_out['index']).copy()

    def init(self, img, bbox):
        x, y, w, h = bbox
        self.center_pos = np.array([x + (w - 1) / 2, y + (h - 1) / 2])
        self.size = np.array([w, h], dtype=np.float32)
        self.channel_average = np.mean(img, axis=(0, 1)).astype(np.uint8)

        w_z = self.size[0] + CONTEXT_AMOUNT * self.size.sum()
        h_z = self.size[1] + CONTEXT_AMOUNT * self.size.sum()
        s_z = round(np.sqrt(w_z * h_z))

        z_crop = get_subwindow(img, self.center_pos, EXEMPLAR_SIZE,
                               s_z, self.channel_average)
        # Backbone expects [1, 255, 255, 3] — pad template to 255x255
        z_255 = cv2.resize(z_crop[0], (INSTANCE_SIZE, INSTANCE_SIZE))
        z_255 = z_255[np.newaxis].astype(np.uint8)

        feat = self._run_backbone(z_255)       # [1, 48, H, W] or [1, H, W, 48]
        # feat is NCHW from onnx2tf: shape (1, 48, 16, 16)
        # take centre 8x8 spatial patch for template
        if feat.shape[1] == 48:                # NCHW
            zf_8x8 = feat[:, :, 4:12, 4:12]   # [1, 48, 8, 8]
            self.zf = zf_8x8.transpose(0,2,3,1).astype(np.int8)  # [1,8,8,48]
        else:                                  # NHWC
            self.zf = feat[:, 4:12, 4:12, :]  # [1, 8, 8, 48]

    def track(self, img):
        w_z = self.size[0] + CONTEXT_AMOUNT * self.size.sum()
        h_z = self.size[1] + CONTEXT_AMOUNT * self.size.sum()
        s_z = np.sqrt(w_z * h_z)
        scale_z = EXEMPLAR_SIZE / s_z
        s_x = s_z * (INSTANCE_SIZE / EXEMPLAR_SIZE)

        x_crop = get_subwindow(img, self.center_pos, INSTANCE_SIZE,
                               round(s_x), self.channel_average)

        # Run backbone on search image
        xf = self._run_backbone(x_crop)        # [1, 48, 16, 16] int8
        if xf.shape[1] == 48:                  # NCHW -> NHWC
            xf = xf.transpose(0, 2, 3, 1)     # [1, 16, 16, 48]

        # Run head
        self.hd_interp.set_tensor(self.hd_zf['index'], self.zf)
        self.hd_interp.set_tensor(self.hd_xf['index'], xf)
        self.hd_interp.invoke()

        cls_int8 = self.hd_interp.get_tensor(self.hd_cls['index'])  # [1,16,16,2]
        loc_int8 = self.hd_interp.get_tensor(self.hd_loc['index'])  # [1,16,16,4]

        # Dequantize
        cls_s, cls_zp = self.hd_cls['quantization']
        loc_s, loc_zp = self.hd_loc['quantization']
        cls_f = dequantize(cls_int8, cls_s, cls_zp)
        loc_f = dequantize(loc_int8, loc_s, loc_zp)

        score = convert_score(cls_f)
        pred_bbox = convert_bbox(loc_f, self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        s_c = change(sz(pred_bbox[2], pred_bbox[3]) /
                     sz(self.size[0] * scale_z, self.size[1] * scale_z))
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2] / pred_bbox[3]))
        penalty = np.exp(-(r_c * s_c - 1) * PENALTY_K)

        pscore = penalty * score
        pscore = pscore * (1 - WINDOW_INFLUENCE) + self.window * WINDOW_INFLUENCE

        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]
        w  = self.size[0] * (1 - lr) + bbox[2] * lr
        h  = self.size[1] * (1 - lr) + bbox[3] * lr
        cx, cy, w, h = bbox_clip(cx, cy, w, h, img.shape)

        self.center_pos = np.array([cx, cy])
        self.size = np.array([w, h])

        return {
            'bbox': [cx - w/2, cy - h/2, w, h],
            'best_score': float(score[best_idx]),
        }


# ---------------------------------------------------------------------------
# Frame source
# ---------------------------------------------------------------------------
def get_frames(source):
    """Yields (frame, fps). fps is None for webcam/image dirs."""
    if str(source) == '0' or (isinstance(source, str) and source.isdigit()):
        cap = cv2.VideoCapture(int(source))
        for _ in range(5): cap.read()
        while True:
            ret, frame = cap.read()
            if ret: yield frame, None
            else: break
    elif isinstance(source, str) and os.path.isdir(source):
        imgs = sorted(glob(os.path.join(source, '*.jp*')),
                      key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        for p in imgs:
            yield cv2.imread(p), None
    else:
        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while True:
            ret, frame = cap.read()
            if ret: yield frame, fps
            else: break
        cap.release()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='NanoTrack TFLite demo (Coral)')
    parser.add_argument('--backbone', default=None,
                        help='override backbone model path '
                             '(default: _edgetpu variant, or _int8 with --no_tpu)')
    parser.add_argument('--head',     default=None,
                        help='override head model path '
                             '(default: _edgetpu variant, or _int8 with --no_tpu)')
    parser.add_argument('--video',    default='./bin/Video.mp4',
                        help='video file, image dir, or 0 for webcam')
    parser.add_argument('--no_tpu',   action='store_true',
                        help='disable Edge TPU delegate (CPU fallback)')
    parser.add_argument('--save',     action='store_true',
                        help='save output video')
    parser.add_argument('--bbox',     type=int, nargs=4, default=None,
                        metavar=('X','Y','W','H'),
                        help='initial bbox (x y w h). Overrides click mode.')
    parser.add_argument('--size',     type=int, default=None,
                        metavar='PX',
                        help='click-to-track mode: bbox side length in pixels. '
                             'Click target on the first frame; bbox is centred there.')
    args = parser.parse_args()

    use_tpu = not args.no_tpu
    backbone_path = args.backbone or (BACKBONE_TPU if use_tpu else BACKBONE_CPU)
    head_path     = args.head     or (HEAD_TPU     if use_tpu else HEAD_CPU)

    print(f"Loading models ({'Edge TPU' if use_tpu else 'CPU'})...")
    print(f"  backbone: {backbone_path}")
    print(f"  head:     {head_path}")
    tracker = NanoTrackerTFLite(backbone_path, head_path, use_tpu=use_tpu)
    print("Models loaded.")

    video_name = os.path.splitext(os.path.basename(str(args.video)))[0]
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)

    writer = None
    first_frame = True

    def click_for_bbox(frame, size):
        """Show frame and wait for a single mouse click; return centred bbox."""
        click = {'pt': None}
        def on_mouse(event, x, y, flags, _):
            if event == cv2.EVENT_LBUTTONDOWN:
                click['pt'] = (x, y)
        cv2.setMouseCallback(video_name, on_mouse)
        prompt = f"Click target ({size}x{size} px). Press q to quit."
        while click['pt'] is None:
            disp = frame.copy()
            cv2.putText(disp, prompt, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow(video_name, disp)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                sys.exit(0)
        cv2.setMouseCallback(video_name, lambda *a, **k: None)
        x, y = click['pt']
        return [x - size // 2, y - size // 2, size, size]

    import time
    frame_period = None
    next_frame_time = None
    for frame, fps in get_frames(args.video):
        if fps and frame_period is None:
            frame_period = 1.0 / fps
            next_frame_time = time.monotonic()
        if first_frame:
            if args.bbox:
                init_rect = args.bbox
            elif args.size:
                init_rect = click_for_bbox(frame, args.size)
            else:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            tracker.init(frame, init_rect)
            first_frame = False

            if args.save:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(
                    f"{video_name}_tracked.mp4",
                    cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h)
                )
            continue

        outputs = tracker.track(frame)
        bbox = list(map(int, outputs['bbox']))
        score = outputs['best_score']

        cv2.rectangle(frame,
                      (bbox[0], bbox[1]),
                      (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                      (0, 255, 0), 2)
        cv2.putText(frame, f"score: {score:.2f}", (bbox[0], bbox[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(video_name, frame)
        if writer:
            writer.write(frame)

        # Pace playback to the source video's FPS
        if frame_period is not None:
            next_frame_time += frame_period
            wait_ms = max(1, int((next_frame_time - time.monotonic()) * 1000))
        else:
            wait_ms = 1
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
