import torch
import torch.nn as nn
from vgg import vgg19
import glob
from PIL import Image, ImageFile  # ImageOps
from io import BytesIO
import requests

import random
import numpy as np

import traceback

from misc import log
import mlsim_lib
import datetime
import tifffile as tiff
import os
import hashlib
import time
import socket

ImageFile.LOAD_TRUNCATED_IMAGES = True

features = []
files = []
model = None
opt = None
device = []


server_socket = None
microManagerPluginState = False



def handle_microManagerPluginState(desiredState, port):
    global microManagerPluginState
    global server_socket

    if desiredState == 'on' and microManagerPluginState == False:
        th = threading.Thread(target=start_plugin_server, args=(port,))
        th.daemon = True
        th.start()
    elif desiredState == 'off' and microManagerPluginState == True:
        server_socket.close()
        microManagerPluginState = False


def set_microManagerPluginState(value):
    global microManagerPluginState
    microManagerPluginState = value






def reconstruct(exportdir,filepaths, conn):
    global model
    global opt

    if model is None:
        opt = mlsim_lib.GetOptions_allRnd()
        model = mlsim_lib.LoadModel(opt)

    os.makedirs(exportdir,exist_ok=True)
    result_arr = []

    log('received filepaths %s' % filepaths)

    for idx, filepath in enumerate(filepaths):
        log('ITRERASTION %d %s' % (idx,filepath))
        # status reconstruction
        conn.send(("siReconstructing,%d,%d" % (idx, len(filepaths))).encode())
        ready = conn.recv(20).decode()

        outfile = '%s/%s' % (exportdir,
                             datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3])
        img = tiff.imread(filepath, key=range(9))
        sr, wf, out = mlsim_lib.EvaluateModel(model, opt, img, outfile)
        result_arr.append(sr)
    
    conn.send("sd".encode())  # status done
    ready = conn.recv(20).decode()

    return result_arr



## --------------------------------------------------------------------------------
##  For Micromanager plugin
## --------------------------------------------------------------------------------

import asyncio
import copy
count = -1
asyncmodels = []
num_models = 5
import threading

class AsyncModel:
    def __init__(self, opt):
        self.resultReady = True
        self.model = mlsim_lib.LoadModel(opt)
        self.result = np.zeros((480,512))
        self.dependency = None

    def getResult(self, opt, img):
        th = threading.Thread(target=mlsim_lib.EvaluateModelRealtimeAsync, args=(self, opt, img))
        th.daemon = True
        th.start()
        while not self.dependency.resultReady:
            time.sleep(0.01)
        self.dependency.resultReady = False
        return self.dependency.result


def reconstruct_image(img):
    global asyncmodels
    global opt
    global count

    if len(asyncmodels) == 0:
        opt = mlsim_lib.GetOptions_allRnd()

        for i in range(num_models):
            asyncmodels.append(AsyncModel(opt))
        for i in range(num_models):
            asyncmodels[i].dependency = asyncmodels[(i+1) % num_models]
    
    count += 1

    return asyncmodels[count % num_models].getResult(opt, img)




import numpy as np
import cv2
from PIL import Image
import pickle
import matplotlib.pyplot as plt
# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore, QtGui

def decodeImage(imgData,w,h,bytesPerPixel,numComponents):

    if numComponents >= 3: # assuming RGB
        if bytesPerPixel == 4: # assuming alpha channel is present - numComponents can be 3 simultaneously with bytesPerPixel being 4
            img = Image.frombuffer("RGBA",(w,h),imgData,"raw","BGRA").convert("RGB")
            return np.array(img) # will be uint8
        else:
            print('image decoder not implemented',w,h,bytesPerPixel,numComponents)
            return None
    elif numComponents == 1:
        data2 = np.frombuffer(imgData,dtype='uint8')
        arr1 = np.reshape(data2[::2][:h*w*numComponents],(h,w,numComponents))
        arr2 = np.reshape(data2[1::2][:h*w*numComponents],(h,w,numComponents))
        comb_arr = 256*arr1 + arr2
        return comb_arr 
    else:
        print('image decoder not implemented',w,h,bytesPerPixel,numComponents)
        return None
    return 

def receiveImageData(conn,npixels,w,h):
    
    bytesstr = None
    count = 0

    while True:
        conn.send('i'.encode())  # send image data
        data = conn.recv(2000)
        if bytesstr is None:
            bytesstr = data
        else:
            bytesstr += data

        count += 1
        # print('%d batch, added now %d' % (count,len(bytesstr)))
        if len(bytesstr) >= npixels:
            conn.send('c'.encode())

            return bytesstr
            ## img = filters.sobel(np.array(img).mean(2))
            

 

def start_plugin_server(port):
    global server_socket
    global microManagerPluginState

    showLiveView = True
    debugMode = False
    host = 'localhost'
    server_socket = socket.socket()  # get instance
    # look closely. The bind() function takes tuple as argument
    server_socket.bind((host, port))  # bind host address and port together
    microManagerPluginState = True

    # configure how many client the server can listen simultaneously
    server_socket.listen(1)
    print('ML-SIM Micromanager now listening for connection')
    try:
        conn, address = server_socket.accept()  # accept new connection
    except:
        conn = None
        errmsg = traceback.format_exc()
        if "not a socket" in errmsg:
            log('Socket closed forcefully')
        else:
            log(errmsg)
        

    if microManagerPluginState and conn is not None:

        stackbuffer = []

        try:
            log('connection from %s' % str(address))

            # receive data stream. it won't accept data packet greater than 2048 bytes  
            count = 0
            t0 = time.perf_counter()
            fps = np.ones((10,))

            if showLiveView and not debugMode:
                plt.figure(figsize=(9,6),frameon=False)
                ax = plt.subplot(111,aspect = 'equal')
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

                # app = QtGui.QApplication([])
                # window = pg.GraphicsView()
                # window.show()
                # window.resize(600,600)
                # window.setWindowTitle('ML-SIM reconstruction')
                # view = pg.ViewBox(enableMouse=True)
                # window.setCentralItem(view)
                # ## lock the aspect ratio
                # view.setAspectLocked(True)
                # view.invertY()
                # ## Create image item
                # imgitem = pg.ImageItem(axisOrder='row-major')
                # view.addItem(imgitem)

                # labelitem = pg.LabelItem()
                # view.addItem(labelitem)

            while microManagerPluginState:
                data = conn.recv(2048).decode()
                vals = data.split(",")
                npixels = int(vals[0])
                w = int(vals[1])
                h = int(vals[2])
                bytesPerPixel = int(vals[3])
                numComponents = int(vals[4])
                # npixels = int.from_bytes(data,"big",signed=True)
                # print("Received",data)
                # print('received npixels',npixels)
                if npixels > 0:
                    imgData = receiveImageData(conn,npixels,w,h)

                    if debugMode:
                        print('received all bytes')
                        pickle.dump(imgData,open('data.pkl','wb'))
                        print('saved as npy')
                    
                    if not debugMode:
                        img = decodeImage(imgData, w,h,bytesPerPixel,numComponents)

                        if img is None:
                            print('Quitting since no valid image provided')
                            break
                        
                        if img.shape[2] == 3: # RGB assumed
                            stackbuffer.extend([img[:,:,0],img[:,:,1],img[:,:,2]])
                        elif img.shape[2] == 1: # assuming scientific camera
                            stackbuffer.append(img[:,:,0])
                        else:
                            print('no implementation for image dimensions')
                            break

                        if len(stackbuffer) == 9:
                            stack = np.array(stackbuffer)
                            stackbuffer = []
                            sr = reconstruct_image(stack)
                            print('obtained reconstructed image',sr.shape)
                        else:
                            print('need more image data, frames in buffer:',len(stackbuffer))
                            continue

                    if showLiveView and not debugMode:
                        plt.cla()
                        plt.gca().imshow(sr,cmap='magma')
                        plt.pause(0.01)
                        
                        # imgitem.setImage(sr)
                        # view.autoRange(padding=0)
                        # pg.QtGui.QApplication.processEvents()
                    
                    # print('received img',img.size)
                    fps[count % 10] = 1 / (time.perf_counter() - t0)
                    count += 1
                    t0 = time.perf_counter()
                    print('img #%d (%dx%d) (%d,%d) - fps: %0.3f' % (count,w,h,bytesPerPixel,numComponents,fps.mean()))
                    continue
                else:
                    print('received',data,'exiting')
                    break
            
        except Exception as e:
            errmsg = traceback.format_exc()
            log(errmsg)
        
        microManagerPluginState = False
        try:
            pg.close()
            pg.win.close()
        except Exception as e:
            errmsg = traceback.format_exc()
            log("Pyqtgraph window not able to close, perhaps not started. Error message is: " + errmsg)
            # send_log(errmsg)
