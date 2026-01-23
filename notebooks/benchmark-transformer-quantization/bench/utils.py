import sys
import platform
import pkg_resources
import torch
import openvino as ov

def check_environment():
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenVINO: {ov.get_version()}")
    print(f"Torch: {torch.__version__}")
    
    # Critical Check: Ensure we have the right extension for NPU/CPU
    available_devices = ov.Core().available_devices
    print(f" Available AI Accelerators: {available_devices}")
    
    if "NPU" in available_devices:
        print(" NPU Detected (Intel Core Ultra/Meteor Lake)")
    elif "GPU" in available_devices:
        print(" iGPU Detected")
    else:
        print(" CPU Only (Performance will be baseline)")