# ML-SIM Graphical Desktop Application

**Author of this repository**:
- Name: Charles Nicklas Christensen
- GitHub: [charlesnchr](http://github.com/charlesnchr)
- Email: <code>charles.n.chr@gmail.com</code>
- Twitter: [charlesnchr](https://twitter.com/charlesnchr)

This sub-project of ML-SIM provides an easy-to-use graphical user interface based on the Electron framework and an engine written in Python to apply the neural network models trained with the ML-SIM method. 

### To-do

- Dropdown model selection to integrate with Python engine. Currently the path for a pre-trained model is hardcoded into `mlsim_lib.py`
- Complete plug-in framework so that other image processing routines can be used
- Support SIM dataset generation within the app based on the simulation code in `../MLSIM_datagen`.
- Implement training routines for generated datasets.

## Installation


### Dependencies
Installation of NodeJS and Yarn are required for building the project. Following version have been tested:
- NodeJS 13.7.0
- Yarn 1.21.1
- Anaconda 3


On Windows, [Chocolatey](https://chocolatey.org/install) is recommended for installation of these dependencies using the following commands in an administrative shell:
```choco install nodejs yarn```

On macOS, [Homebrew](https://brew.sh/) can be used:
```brew install yarn node```

Similarly easy install commands tend to be available on Linux distributions, e.g. with *apt-get* on Ubuntu.

### Installing the project
Open a terminal and go to the root of the project. Then run the command

```yarn``` 

Download required Pytorch models:

```yarn download-models```

The model can also be downloaded explicitly, but the above script is recommended for proper folder placement. [Download link](https://ml-sim.s3.eu-west-2.amazonaws.com/pdist/models/DIV2K_randomised_3x3_20200317.pth).

#### Setup Python environment
First download and install Anaconda if you do not have it: https://www.anaconda.com/products/individual

```
conda create -n ML-SIM python=3.6
```
Make sure to do "conda init" on a fresh install of Anaconda for environment switching to be possible. Now run:
```
conda activate ML-SIM
```
Install the Python module pipenv to manage the virtual Python environment and install all packages from the Pipfile
```
pip install pipenv
pipenv install
```
As of now it is not possible to install packages with the Pipfile by specifying a link via ```-f```, so in order to setup Pytorch, use the following command
```
pipenv run pip install torch==1.8.1+cu102 -f https://download.pytorch.org/whl/torch_stable.html
```
or in macOS and Linux simply
```
pipenv run pip install torch
```

## Run the app
First you start the Python engine with (assuming current directory is as above)
```
yarn socket
```
Assuming everything has initialised well, you should see a terminal print line saying **now listening**. You are now ready to start the graphical user interface with:
```
yarn start
``` 
which will start Electron.
