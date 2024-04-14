import os

c.KernelGatewayApp.ip = "*"
c.KernelGatewayApp.port = 8080

c.KernelGatewayApp.env_process_whitelist = [
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "NSS_WRAPPER_PASSWD",
    "NSS_WRAPPER_GROUP",
]

image_config_file = "/opt/app-root/src/.jupyter/jupyter_kernel_gateway_config.py"

if os.path.exists(image_config_file):
    with open(image_config_file) as fp:
        exec(compile(fp.read(), image_config_file, "exec"), globals())
