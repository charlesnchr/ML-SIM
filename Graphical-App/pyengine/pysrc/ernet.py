import os
import datetime
import math

import torch
import time 

import torch.nn as nn
from PIL import Image, ImageFile  # ImageOps
from skimage import io,exposure,img_as_ubyte
import glob

import numpy as np
import os

from argparse import Namespace

import torchvision.transforms as transforms
toTensor = transforms.ToTensor()  
toPIL = transforms.ToPILImage()      


# ----------------------------------- RCAN ------------------------------------------

def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, opt): 
        super(RCAN, self).__init__()
        n_resgroups = opt.n_resgroups
        n_resblocks = opt.n_resblocks
        n_feats = opt.n_feats
        kernel_size = 3
        reduction = opt.reduction
        act = nn.ReLU(True)
        self.narch = opt.narch
        
        if not opt.norm == None:
            self.normalize, self.unnormalize = normalizationTransforms(opt.norm)
        else:
            self.normalize, self.unnormalize = None, None


        # define head module
        if self.narch == 0:
            modules_head = [conv(opt.nch_in, n_feats, kernel_size)]
            self.head = nn.Sequential(*modules_head)
        else:
            self.head0 = conv(1, n_feats, kernel_size)
            self.head02 = conv(n_feats, n_feats, kernel_size)
            self.head1 = conv(1, n_feats, kernel_size)
            self.head12 = conv(n_feats, n_feats, kernel_size)
            self.head2 = conv(1, n_feats, kernel_size)
            self.head22 = conv(n_feats, n_feats, kernel_size)
            self.head3 = conv(1, n_feats, kernel_size)
            self.head32 = conv(n_feats, n_feats, kernel_size)
            self.head4 = conv(1, n_feats, kernel_size)
            self.head42 = conv(n_feats, n_feats, kernel_size)
            self.head5 = conv(1, n_feats, kernel_size)
            self.head52 = conv(n_feats, n_feats, kernel_size)
            self.head6 = conv(1, n_feats, kernel_size)
            self.head62 = conv(n_feats, n_feats, kernel_size)
            self.head7 = conv(1, n_feats, kernel_size)
            self.head72 = conv(n_feats, n_feats, kernel_size)
            self.head8 = conv(1, n_feats, kernel_size)
            self.head82 = conv(n_feats, n_feats, kernel_size)
            self.combineHead = conv(9*n_feats, n_feats, kernel_size)

            

        # define body module
        modules_body = [
            ResidualGroup(
                conv, n_feats, kernel_size, reduction, act=act, res_scale=1, n_resblocks=n_resblocks) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        if opt.scale == 1:
            modules_tail = [nn.Conv2d(n_feats, opt.nch_out, 1)]
        else:
            modules_tail = [
                Upsampler(conv, opt.scale, n_feats, act=False),
                conv(n_feats, opt.nch_out, kernel_size)]
        
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        if not self.normalize == None:
            x = self.normalize(x)

        if self.narch == 0:
            x = self.head(x)
        else:
            x0 = self.head02(self.head0(x[:,0:0+1,:,:]))
            x1 = self.head12(self.head1(x[:,1:1+1,:,:]))
            x2 = self.head22(self.head2(x[:,2:2+1,:,:]))
            x3 = self.head32(self.head3(x[:,3:3+1,:,:]))
            x4 = self.head42(self.head4(x[:,4:4+1,:,:]))
            x5 = self.head52(self.head5(x[:,5:5+1,:,:]))
            x6 = self.head62(self.head6(x[:,6:6+1,:,:]))
            x7 = self.head72(self.head7(x[:,7:7+1,:,:]))
            x8 = self.head82(self.head8(x[:,8:8+1,:,:]))
            x = torch.cat((x0,x1,x2,x3,x4,x5,x6,x7,x8), 1)
            x = self.combineHead(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        if not self.unnormalize == None:
            x = self.unnormalize(x)

        return x 



# from plotting import testAndMakeCombinedPlots


def remove_dataparallel_wrapper(state_dict):
	r"""Converts a DataParallel model to a normal one by removing the "module."
	wrapper in the module dictionary

	Args:
		state_dict: a torch.nn.DataParallel state dictionary
	"""
	from collections import OrderedDict

	new_state_dict = OrderedDict()
	for k, vl in state_dict.items():
		name = k[7:] # remove 'module.' of DataParallel
		new_state_dict[name] = vl

	return new_state_dict


def changeColour(I): # change colours (used to match WEKA output, request by Meng)
    Inew = np.zeros(I.shape + (3,)).astype('uint8')
    for rowidx in range(I.shape[0]):
        for colidx in range(I.shape[1]):
            if I[rowidx][colidx] == 0:
                Inew[rowidx][colidx] = [198,118,255]
            elif I[rowidx][colidx] == 127:
                Inew[rowidx][colidx] = [79,255,130]
            elif I[rowidx][colidx] == 255:
                Inew[rowidx][colidx] = [255,0,0]
    return Inew



def AssembleStacks(basefolder):
    # export to tif
    
    folders = []
    folders.append(basefolder + '/in')
    folders.append(basefolder + '/out') 

    for subfolder in ['in','out']:
        folder = basefolder + '/' + subfolder
        if not os.path.isdir(folder): continue
        imgs = glob.glob(folder + '/*.jpg')
        imgs.extend(glob.glob(folder + '/*.png'))
        n = len(imgs)
        
        shape = io.imread(imgs[0]).shape
        h = shape[0]
        w = shape[1]
        
        if len(shape) == 2:
            I = np.zeros((n,h,w),dtype='uint8')
        else:
            c = shape[2]
            I = np.zeros((n,h,w,c),dtype='uint8')
        
        for nidx, imgfile in enumerate(imgs):
            img = io.imread(imgfile)
            I[nidx] = img

            print('%s : [%d/%d] loaded imgs' % (folder,nidx+1,len(imgs)),end='\r')
        print('')
        
        stackname = os.path.basename(basefolder)
        stackfilename = '%s/%s_%s.tif' % (basefolder,stackname,subfolder)
        io.imsave(stackfilename,I,compress=6)
        print('saved stack: %s' % stackfilename)



def processImage(net,opt,imgfile,img,savepath_in,savepath_out,idxstr):

    imageSize = opt.imageSize

    h,w = img.shape[0], img.shape[1]
    if imageSize == 0:
        imageSize = 250
        while imageSize+250 < h and imageSize+250 < w:
            imageSize += 250
        print('Set imageSize to',imageSize)

    device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')

    # img_norm = (img - np.min(img)) / (np.max(img) - np.min(img)) 
    images = []

    images.append(img[:imageSize,:imageSize])
    images.append(img[h-imageSize:,:imageSize])
    images.append(img[:imageSize,w-imageSize:])
    images.append(img[h-imageSize:,w-imageSize:])

    proc_images = []
    for idx,sub_img in enumerate(images):
        # sub_img = (sub_img - np.min(sub_img)) / (np.max(sub_img) - np.min(sub_img))
        pil_sub_img = Image.fromarray((sub_img*255).astype('uint8'))
        
        # sub_tensor = torch.from_numpy(np.array(pil_sub_img)/255).float().unsqueeze(0)
        sub_tensor = toTensor(pil_sub_img)

        # print(my_sub_tensor.shape,sub_tensor.shape)

        sub_tensor = sub_tensor.unsqueeze(0)

        with torch.no_grad():
            sr = net(sub_tensor.to(device))
            
            sr = sr.cpu()
            # sr = torch.clamp(sr,min=0,max=1)

            m = nn.LogSoftmax(dim=0)
            sr = m(sr[0])
            sr = sr.argmax(dim=0, keepdim=True)
            
            # pil_sr_img = Image.fromarray((255*(sr.float() / (opt.nch_out - 1)).squeeze().numpy()).astype('uint8'))
            pil_sr_img = toPIL(sr.float() / (opt.nch_out - 1))

            # pil_sr_img.save(opt.out + '/segmeneted_output_' + str(i) + '_' + str(idx) + '.png')
            # pil_sub_img.save(opt.out + '/imageinput_' + str(i) + '_' + str(idx) + '.png')

            proc_images.append(pil_sr_img)
        
    # stitch together
    img1 = proc_images[0]
    img2 = proc_images[1]
    img3 = proc_images[2]
    img4 = proc_images[3]

    woffset = (2*imageSize-w) // 2
    hoffset = (2*imageSize-h) // 2

    img1 = np.array(img1)[:imageSize-hoffset,:imageSize-woffset]
    img3 = np.array(img3)[:imageSize-hoffset,woffset:]
    top = np.concatenate((img1,img3),axis=1)

    img2 = np.array(img2)[hoffset:,:imageSize-woffset]
    img4 = np.array(img4)[hoffset:,woffset:]
    bot = np.concatenate((img2,img4),axis=1)

    oimg = np.concatenate((top,bot),axis=0)
    # crop?
    # oimg = oimg[10:-10,10:-10]
    # img = img[10:-10,10:-10]
    # remove boundaries? 
    # oimg[:10,:] = 0
    # oimg[-10:,:] = 0
    # oimg[:,:10] = 0
    # oimg[:,-10:] = 0

    if opt.stats_tubule_sheet:
        fraction1 = np.sum(oimg == 255) # tubule
        fraction2 = np.sum(oimg == 127) # sheet
        npix = w*h
        opt.csvfid.write('%s:%s,%0.4f,%0.4f,%0.4f\n' % (os.path.basename(imgfile),idxstr,fraction1/npix,fraction2/npix,fraction1/fraction2))
    if opt.weka_colours:
        oimg = changeColour(oimg)

    Image.fromarray(oimg).save(savepath_out)
    if opt.save_input:
        io.imsave(savepath_in,img_as_ubyte(img))
        
    # Image.fromarray((img*255).astype('uint8')).save('%s/input_%04d.png' % (opt.out,i))



def EvaluateModel(opt,conn,graph_metrics=False):

    if opt.stats_tubule_sheet:
        # if opt.out == 'root':
        #     if opt.root[0].lower() in ['jpg','png','tif']:
        #         pardir = os.path.abspath(os.path.join(opt.root,os.pardir))
        #         opt.csvfid = open('%s/stats_tubule_sheet.csv' % pardir,'w')
        #     else:
        #         opt.csvfid = open('%s/stats_tubule_sheet.csv' % opt.root,'w')
        # else:
        #     opt.csvfid = open('%s/stats_tubule_sheet.csv' % opt.out,'w')
        opt.csvfid.write('Filename,Tubule fraction,Sheet fraction,Tubule/sheet ratio\n')

    net = RCAN(opt)
    device = torch.device('cuda' if torch.cuda.is_available() and not opt.cpu else 'cpu')
    net.to(device)

    checkpoint = torch.load(opt.weights, map_location=device)
    print('loading checkpoint',opt.weights)
    net.load_state_dict(checkpoint['state_dict'])

    if opt.root[0].split('.')[-1].lower() in ['png','jpg','tif']:
        imgs = opt.root
    else:
        imgs = []
        for ext in opt.ext:
            # imgs.extend(glob.glob(opt.root + '/*.jpg')) # scan only folder
            if len(imgs) == 0: # scan everything
                for dir in opt.root:
                    imgs.extend(glob.glob(dir + '/**/*.%s' % ext,recursive=True))


    # find total number of images to process
    nimgs = 0
    for imgidx, imgfile in enumerate(imgs):
        basepath, ext = os.path.splitext(imgfile)

        if ext.lower() == '.tif':
            img = io.imread(imgfile)
            if len(img.shape) == 2: # grayscale 
                nimgs += 1
            elif  img.shape[2] <= 3:
                nimgs += 1
            else: # t or z stack
                nimgs += img.shape[0]
        else:
            nimgs += 1

    outpaths = []
    imgcount = 0

    for imgidx, imgfile in enumerate(imgs):
        basepath, ext = os.path.splitext(imgfile)

        if ext.lower() == '.tif':
            img = io.imread(imgfile).astype('float32')
        else:
            img = np.array(Image.open(imgfile)).astype('float32')

        if len(img.shape) > 2 and img.shape[2] <= 3:
            print('removing colour channel')
            img = np.max(img,2) # remove colour channel

        # img = io.imread(imgfile)
        # img = (img - np.min(img)) / (np.max(img) - np.min(img)) 

        # filenames for saving
        idxstr = '%04d' % imgidx
        if opt.out == 'root': # save next to orignal
            savepath_out = imgfile.replace(ext,'_out_' + idxstr + '.png')
            savepath_in = imgfile.replace(ext,'_in_' + idxstr + '.png')
            basesavepath_graphfigures = imgfile.replace(ext,'')
        else: 
            pass # not implemented

        # process image
        if len(img.shape) == 2:            
            # status
            conn.send(("siSegmentation,%d,%d" % (imgcount, nimgs)).encode())
            ready = conn.recv(20).decode()

            p1,p99 = np.percentile(img,1),np.percentile(img,99)
            print(img.shape,np.max(img),np.min(img))
            imgnorm = exposure.rescale_intensity(img,in_range=(p1,p99))
            print(imgnorm.shape,np.max(imgnorm),np.min(imgnorm))
            processImage(net,opt,imgfile,imgnorm,savepath_in,savepath_out,idxstr)
            
            # send result    
            outpaths.append(savepath_out)
            conn.send(('2' + '\n'.join(outpaths)).encode()) # partial render
            ready = conn.recv(20).decode()

            # graph processing
            if graph_metrics:
                conn.send(("siGraph metrics,%d,%d" % (imgcount, nimgs)).encode())
                ready = conn.recv(20).decode()

                graph_out_paths = performGraphProcessing(savepath_out,opt, conn, basesavepath_graphfigures)

                outpaths.extend(graph_out_paths)
                conn.send(('2' + '\n'.join(outpaths)).encode()) # partial render
                ready = conn.recv(20).decode()

            imgcount += 1

        else: # more than 3 channels, assuming stack
            basefolder = basepath
            os.makedirs(basefolder,exist_ok=True)
            if opt.save_input:
                os.makedirs(basefolder + '/in',exist_ok=True)
            if graph_metrics:
                os.makedirs(basefolder + '/graph',exist_ok=True)
            os.makedirs(basefolder + '/out',exist_ok=True)

            for subimgidx in range(img.shape[0]):
                # status
                conn.send(("siSegmentation,%d,%d" % (imgcount, nimgs)).encode())
                ready = conn.recv(20).decode()

                idxstr = '%04d_%04d' % (imgidx,subimgidx)
                savepath_in = '%s/in/%s.png' % (basefolder,idxstr)
                savepath_out = '%s/out/%s.png' % (basefolder,idxstr)
                basesavepath_graphfigures = '%s/graph/%s' % (basefolder,idxstr)
                p1,p99 = np.percentile(img[subimgidx],1),np.percentile(img[subimgidx],99)
                imgnorm = exposure.rescale_intensity(img[subimgidx],in_range=(p1,p99))
                processImage(net,opt,imgfile,imgnorm,savepath_in,savepath_out,idxstr)

                # send result                
                outpaths.append(savepath_out)
                conn.send(('2' + '\n'.join(outpaths)).encode()) # partial render
                ready = conn.recv(20).decode()

                # graph processing
                if graph_metrics:
                    conn.send(("siGraph metrics,%d,%d" % (imgcount, nimgs)).encode())
                    ready = conn.recv(20).decode()

                    graph_out_paths = performGraphProcessing(savepath_out,opt, conn, basesavepath_graphfigures)

                    outpaths.extend(graph_out_paths)
                    conn.send(('2' + '\n'.join(outpaths)).encode()) # partial render
                    ready = conn.recv(20).decode()

                imgcount += 1
            AssembleStacks(basefolder)


    if opt.stats_tubule_sheet:
        opt.csvfid.close()

    conn.send("sd".encode())  # status done
    ready = conn.recv(20).decode()
    print(outpaths)
    return outpaths


import cv2
import sknw
from skimage.morphology import skeletonize

def remove_isolated_pixels(image):
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for label in range(num_stats):
        if stats[label,cv2.CC_STAT_AREA] < 50:
            new_image[labels == label] = 0

    return new_image


def binariseImage(I):
    ind = I[:,:,0] > 250
    Ibin = np.zeros((I.shape[0],I.shape[1])).astype('uint8')
    Ibin[ind] = 255
    Ibin = remove_isolated_pixels(Ibin)
    return Ibin


def getGraph(img,basename):
    img = binariseImage(img) / 255
    ske = skeletonize(img).astype(np.uint16)
    # ske = img.astype('uint16')

    # build graph from skeleton
    graph = sknw.build_sknw(ske)

    # draw edges by pts
    for (s,e) in graph.edges():
        ps = graph[s][e]['pts']
        
    # draw node by o
    nodes = graph.nodes()
    ps = np.array([nodes[i]['o'] for i in nodes])   

    # print('EDGES',graph.edges())
    # print('NODES',graph.nodes())
    
    open('%s_edges.dat' % basename,'w').write(str(graph.edges()))
    open('%s_nodes.dat' % basename,'w').write(str(graph.nodes()))

    edges = np.array(graph.edges())

    return edges, nodes

import networkx as nx
import json
import matplotlib.pyplot as plt
import collections

plt.switch_backend('agg')


#build teh networkx graph from node and egdes lists 
def build_graph(nodes,edges):
    G=nx.Graph()
    G.add_nodes_from(nodes)
    for edge in edges:
        G.add_edge(edge[0],edge[1])
    return G

# perform some analysis 
def simple_analysis(G):
    no_nodes=G.number_of_nodes()
    no_edges=G.number_of_edges()
    assortativity = nx.degree_assortativity_coefficient(G)
    clustering = nx.average_clustering(G)
    compo = nx.number_connected_components(G)
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    size_G0_edges = G0.number_of_edges()
    size_G0_nodes = G0.number_of_nodes()
    ratio_nodes=size_G0_nodes/no_nodes
    ratio_edges =size_G0_edges/no_edges
    return no_nodes,no_edges,assortativity, clustering, compo, ratio_nodes, ratio_edges

# here we generate a histogram of degrees with a specified colour
def degree_histogram(savepath,G,colour='blue'):
    plt.figure()

    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    fig1, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color=colour)
    plt.ylabel("Count")
    plt.xlabel("%Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.savefig(savepath)

    plt.close()

# here we generate an image where nodes are coloured according to their degrees 
def graph_image(savepath,G):
    plt.figure()
    node_color = [float(G.degree(v)) for v in G]
    nx.draw_spring(G,node_size=10,node_color=node_color)
    plt.savefig(savepath)

    plt.close()



def performGraphProcessing(imgfile,opt, conn, basename):
    
    savepath_hist = '%s_hist.png' % basename
    savepath_graph = '%s_graph.png' % basename

    img = io.imread(imgfile)
    edges, nodes = getGraph(img, basename)
    G=build_graph(nodes,edges)
    simple_analysis(G)
    degree_histogram(savepath_hist,G,'goldenrod')
    graph_image(savepath_graph,G)

    plt.close()

    return [savepath_graph,savepath_hist]



def segment(exportdir,filepaths,conn,weka_colours,stats_tubule_sheet,graph_metrics,save_in_original_folders,save_input=True):
    import sys
    opt = Namespace()
    opt.root = filepaths
    opt.ext = ['jpg','png','tif']
    opt.stats_tubule_sheet = stats_tubule_sheet
    opt.weka_colours = weka_colours
    opt.save_input = save_input
    
    opt.exportdir = exportdir
    opt.jobname = datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')[:-3]

    if stats_tubule_sheet:
        csvfid_path = '%s/%s_stats_tubule_sheet.csv' % (opt.exportdir, opt.jobname)
        opt.csvfid = open(csvfid_path,'w')

    ## model specific 
    opt.imageSize = 1000
    opt.n_resblocks = 10
    opt.n_resgroups = 2
    opt.n_feats = 64
    opt.reduction = 16
    opt.narch = 0
    opt.norm = None
    opt.nch_in = 1
    opt.nch_out = 3
    opt.cpu = False
    opt.weights = '../models/meng_3colours_4.pth'
    opt.scale = 1
    
    
    if save_in_original_folders:
        opt.out = "root"

    print(opt)
    
    if graph_metrics:
        return EvaluateModel(opt,conn,graph_metrics)
    else:
        return EvaluateModel(opt,conn)



