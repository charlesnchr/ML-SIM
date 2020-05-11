# ML-SIM

## Status of repository
The code used for the preprint publication is currently being polished and commented to make it more easy to read. Please look back soon to find the code.

**ML-SIM: A deep neural network for the reconstruction of structured illumination microscopy images**

_Charles N. Christensen<sup>1,2,*</sup>, Edward N. Ward<sup>1</sup>, Pietro Lio<sup>2</sup>, Clemens F. Kaminski_</br></br>
<sup>1</sup>University of Cambridge, Department of Chemical Engineering and Biotechnology, Laser Analytics Group</br>
<sup>2</sup>University of Cambridge, Department of Computer Science and Technology, Artificial Intelligence Group</br>
<sup> *</sup>Author of this repository - GitHub username: [charlesnchr](http://github.com/charlesnchr) - Email address: <code>charles.n.chr@gmail.com</code>

## Read the preprint
Preprint paper on arXiv [https://arxiv.org/abs/2003.11064](https://arxiv.org/abs/2003.11064) 


## Web app to test the model
There is an online browser-based ready-to-use implementation available at:
[http://ML-SIM.com](http://ML-SIM.com). The model used in this tool assumes that the inputs are 9 frame SIM stacks of 512x512 resolution; i.e. 3 orientations and 3 phase shifts. It will work for other dimensions, but is unlikely to be good.

## Desktop app
An easy to install and use desktop app for Windows 10, macOS and Linux will be available soon. The program will allow one to batch process a set of directories including subdirectories that contain TIFF stacks, and customise and select the model that is used for reconstruction. See screenshot below.

<img src="fig/screenshot 20200511.jpg">
