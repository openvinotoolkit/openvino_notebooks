# Facies segmentation Python Demo

This demo demonstrate how to run facies classification using OpenVINO&trade;

This model came from seismic interpretation tasks. Fasies is the overall characteristics of a rock unit that reflect its origin and differentiate the unit from others around it.  Mineralogy and sedimentary source, fossil content, sedimentary structures and texture distinguish one facies from another. Data are presented in the 3D arrays.


## How It Works
Upon the start-up, the demo application loads a network and an given dataset file to the Inference Engine plugin. When inference is done, the application displays 3d itkwidget viewer with facies interpretation.

## Running

### Installation of dependencies
First of all, you need to create the required CPU extension, just follow the instructions below:

Steps to create CPU extension:

```bash
$ source <openvino_install>/bin/setupvars.sh
$ export TBB_DIR=${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/external/tbb/cmake/

$ cd ./user_ie_extensions
$ mkdir build && cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc --all)
$ mv libunpool_cpu_extension.so ../../
```
### Setup virtual-env

Step 1: Install [pyenv](https://github.com/pyenv/pyenv)
```sh
$ curl -L https://github.com/pyenv/pyenv-installer/raw/master/bin/pyenv-installer | bash
```

Step 2 Install python 3.6.9:
```sh
$ pyenv install 3.6.9
```
NOTE: If you get an error when trying to install python, run the following command:
```sh
$ export PATH="$HOME/.pyenv/bin:$PATH"
```

Step 3: Create virtual-env in this directory:
```sh
$ cd notebooks/202-facies-segmentation
$ ~/.pyenv/versions/3.6.8/bin/python -m venv facies_demo_env
```

Step 4: Activate env:
```sh
$ source facies_demo_env/bin/activate 
```

Step 5: add your env to jupyter:
```sh
$ ipython kernel install --name "facies_demo_env" --user
```

Step 6 (Optional): You can install the required packages now with the following command, or later, inside the jupyter demo notebook:
```sh
$ pip install -r requirements.txt
```

### Download model:

Step 1: Create model folder and download a model:

```sh
$ cd notebooks/202-facies-segmentation
$ mkdir model && cd model
$ wget -O facies-segmentation-deconvnet.bin https://www.dropbox.com/s/x0c7ao8kebxykj1/facies-segmentation-deconvnet.bin?dl=1 
$ wget -O facies-segmentation-deconvnet.xml https://www.dropbox.com/s/g288xdcd7xumqm7/facies-segmentation-deconvnet.xml?dl=1
$ wget -O facies-segmentation-deconvnet.mapping https://www.dropbox.com/s/a7kge25hfpjnhvf/facies-segmentation-deconvnet.mapping?dl=1
```

### Run notebook

Run Jupyter notebook with demo
```bash
$ cd notebooks/202-facies-segmentation
$ jupyter notebook
```

## Demo Output

The application uses Jupyter notebook to display 3d itkwidget with resulting instance classification masks.

<img src="demo.png"
     alt="Markdown Monster icon"
     style="margin:0 auto; display: block"/>
