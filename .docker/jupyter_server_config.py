import os

port = int(os.environ.get("JUPYTER_NOTEBOOK_PORT", "8080"))

c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = port
c.ServerApp.open_browser = False
c.ServerApp.quit_button = False

if os.environ.get("JUPYTERHUB_SERVICE_PREFIX"):
    c.ServerApp.base_url = os.environ.get("JUPYTERHUB_SERVICE_PREFIX")

password = os.environ.get("JUPYTER_NOTEBOOK_PASSWORD")
if password:
    import notebook.auth

    c.ServerApp.password = notebook.auth.passwd(password)
    del password
    del os.environ["JUPYTER_NOTEBOOK_PASSWORD"]

image_config_file = "/opt/app-root/src/.jupyter/jupyter_server_config.py"

if os.path.exists(image_config_file):
    with open(image_config_file) as fp:
        exec(compile(fp.read(), image_config_file, "exec"), globals())
