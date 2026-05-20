# Copyright (c) SenseTime. All Rights Reserved.
import argparse
import os
import torch
import numpy
import torch
import sys
import onnx
from onnxsim import simplify

# Set path for local imports
sys.path.append(os.getcwd())

from nanotrack.core.config import cfg
from nanotrack.utils.model_load import load_pretrain
from nanotrack.models.model_builder import ModelBuilder

try:
    _np_scalar = numpy._core.multiarray.scalar   # numpy >= 2.0
except AttributeError:
    _np_scalar = numpy.core.multiarray.scalar    # numpy < 2.0
torch.serialization.add_safe_globals([_np_scalar])

# Force CPU to avoid CUDA-specific artifacts in the ONNX graph
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description='NanoTrack TPU Export')
parser.add_argument('--config', type=str, default='./models/config/configv3.yaml', help='config file')
parser.add_argument('--snapshot', default='models/pretrained/nanotrackv2.pth', type=str, help='snapshot models to eval')
args = parser.parse_args()

def export_and_simplify(model_part, dummy_input, output_path, input_names, output_names):
    """Exports to ONNX and immediately runs the simplifier."""
    print(f"--- Exporting {output_path} ---")
    
    # Export with Opset 11 for better TFLite/TPU compatibility
    torch.onnx.export(
        model_part,
        dummy_input,
        output_path,
        input_names=input_names,
        output_names=output_names,
        verbose=False,
        opset_version=18,
        do_constant_folding=True
    )
    
    # Simplify the model
    onnx_model = onnx.load(output_path)
    model_simp, check = simplify(onnx_model)
    assert check, f"Simplified ONNX check failed for {output_path}"
    onnx.save(model_simp, output_path.replace(".onnx", "_sim.onnx"))
    print(f"Successfully simplified: {output_path.replace('.onnx', '_sim.onnx')}")

def main():
    # 1. Load Configuration and Model
    cfg.merge_from_file(args.config)
    model = ModelBuilder()
    
    # 2. Load Weights to CPU
    device = torch.device('cpu')
    model = load_pretrain(model, args.snapshot)
    model.eval().to(device)

    # 3. Create Output Directory
    os.makedirs('./models/onnx', exist_ok=True)

    # --- PART A: BACKBONE (Feature Extractor) ---
    # Shape [Batch, Channels, Height, Width]
    # Search image is typically 255x255
    backbone_net = model.backbone
    backbone_input = torch.randn([1, 3, 255, 255], device=device)
    
    export_and_simplify(
        backbone_net, 
        backbone_input, 
        './models/onnx/nanotrack_backbone.onnx',
        ['input'], 
        ['output']
    )

    # --- PART B: HEAD (Classification & Regression) ---
    # Backbone outputs 48ch; template 127x127 -> 8x8, search 255x255 -> 16x16
    head_net = model.ban_head
    head_zf = torch.randn([1, 48, 8, 8], device=device)
    head_xf = torch.randn([1, 48, 16, 16], device=device)
    
    export_and_simplify(
        head_net,
        (head_zf, head_xf),
        './models/onnx/nanotrack_head.onnx',
        ['zf', 'xf'],
        ['cls', 'reg']
    )

if __name__ == '__main__':
    main()