"""Build TensorRT engine from ONNX model using Python."""

import tensorrt as trt
import sys
from pathlib import Path

def build_engine(onnx_path: str, engine_path: str):
    """Build TensorRT engine from ONNX model."""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    print(f"Loading ONNX file: {onnx_path}")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    # Build config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB
    config.set_flag(trt.BuilderFlag.FP16)
    
    # Build engine
    print("Building TensorRT engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Failed to build engine")
        return False
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"Engine saved to: {engine_path}")
    return True

if __name__ == "__main__":
    onnx_file = "models/yolov8n.onnx"
    engine_file = "models/yolov8n_fp16.engine"
    
    if not Path(onnx_file).exists():
        print(f"Error: {onnx_file} not found!")
        sys.exit(1)
    
    success = build_engine(onnx_file, engine_file)
    sys.exit(0 if success else 1)


