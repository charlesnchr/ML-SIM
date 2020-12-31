import os
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import functools # used by RRDBNet
import math


def GetModel(opt):
    if opt.model.lower() == 'edsr':
        net = EDSR(opt)
    elif opt.model.lower() == 'rcan':
        net = RCAN(opt)
    elif opt.model.lower() == 'rnan':
        net = RNAN(opt)
    elif opt.model.lower() == 'rrdb' or opt.model.lower() == 'esrgan':
        net = GeneratorRRDB(opt, filters=opt.n_feats, num_res_blocks=opt.n_resblocks,num_upsample=0)
    elif opt.model.lower() == 'srresnet' or opt.model.lower() == 'srgan':
        net = Generator(16, opt)
    elif opt.model.lower() == 'unet':        
        net = UNet(opt.nch_in,opt.nch_out,opt)
    elif opt.model.lower() == 'unet_n2n':        
        net = UNet_n2n(opt.nch_in,opt.nch_out,opt)
    elif opt.model.lower() == 'unet60m':        
        net = UNet60M(opt.nch_in,opt.nch_out)
    elif opt.model.lower() == 'unetrep':        
        net = UNetRep(opt.nch_in,opt.nch_out)        
    elif opt.model.lower() == 'unetgreedy':        
        net = UNetGreedy(opt.nch_in,opt.nch_out)        
    elif opt.model.lower() == 'mlpnet':        
        net = MLPNet()                
    elif opt.model.lower() == 'ffdnet':        
        net = FFDNet(opt.nch_in)
    elif opt.model.lower() == 'dncnn':        
        net = DNCNN(opt.nch_in)
    elif opt.model.lower() == 'fouriernet':        
        net = FourierNet()        
    elif opt.model.lower() == 'fourierconvnet':        
        net = FourierConvNet()    
    elif opt.model.lower() == 'vgg':
        net = models.vgg19(pretrained=True)
        # OUT_FEATURES = net.classifier[0].out_features
        # first_fc = nn.Linear(OUT_FEATURES, opt.nch_in)]
        # net.classifier[0] = first_fc
        IN_FEATURES = net.classifier[-1].in_features
        final_fc = nn.Linear(IN_FEATURES, opt.nch_out)

        # freeze all layers
        for child in net.children():
            for param in child.parameters():
                param.requires_grad = False
        
        # except last
        net.classifier[-1] = final_fc
        for param in net.classifier[-1].parameters():
            param.requires_grad = True
    else:
        print("model undefined")    
        return None
    
    if not opt.cpu:
        net.cuda()
        net = nn.DataParallel(net)

    return net
    


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean, rgb_std, sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False



def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)



class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []

        m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        m.append(nn.ReLU(True))

        m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
        
        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res



class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(self,opt):
        super(EDSR, self).__init__()

        n_resblocks = opt.n_resblocks # originally 16
        n_feats = opt.n_feats # originally 64
        kernel_size = 3
        act = nn.ReLU(True)
        
        # define head module
        m_head = [conv(opt.nch_in, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=0.1
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        if opt.scale == 1:
            if opt.task == 'segment':
                m_tail = [nn.Conv2d(n_feats, 2, 1)]
            else:
                m_tail = [conv(n_feats, opt.nch_out, kernel_size)]
        else:
            m_tail = [
                Upsampler(conv, opt.scale, n_feats, act=False),
                conv(n_feats, opt.nch_out, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x 



# ----------------------------------- RCAN ------------------------------------------

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
            if opt.task == 'segment':
                modules_tail = [nn.Conv2d(n_feats, opt.nch_out, 1)]
            else:
                modules_tail = [conv(n_feats, opt.nch_out, kernel_size)]
        else:
            modules_tail = [
                Upsampler(conv, opt.scale, n_feats, act=False),
                conv(n_feats, opt.nch_out, kernel_size)]
        
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

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

        return x 





# ----------------------------------- RNAN ------------------------------------------


# add NonLocalBlock2D
# reference: https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_simple_version.py
class NonLocalBlock2D(nn.Module):
    def __init__(self, in_channels, inter_channels):
        super(NonLocalBlock2D, self).__init__()
        
        self.in_channels = in_channels
        self.inter_channels = inter_channels
        
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        # for pytorch 0.3.1
        #nn.init.constant(self.W.weight, 0)
        #nn.init.constant(self.W.bias, 0)
        # for pytorch 0.4.0
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)
        
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):

        batch_size = x.size(0)
        
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        
        g_x = g_x.permute(0,2,1)
        
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        
        theta_x = theta_x.permute(0,2,1)
        
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        
        f = torch.matmul(theta_x, phi_x)
       
        f_div_C = F.softmax(f, dim=1)
        
        
        y = torch.matmul(f_div_C, g_x)
        
        y = y.permute(0,2,1).contiguous()
         
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


## define trunk branch
class TrunkBranch(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(TrunkBranch, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        self.body = nn.Sequential(*modules_body)
    
    def forward(self, x):
        tx = self.body(x)

        return tx



## define mask branch
class MaskBranchDownUp(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(MaskBranchDownUp, self).__init__()
        
        MB_RB1 = []
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
         
        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat,n_feat, 3, stride=2, padding=1))
        
        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
         
        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat,n_feat, 6, stride=2, padding=2))   
        
        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat,n_feat, 1, padding=0, bias=True))
       
        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up  = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)
    
    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx

## define nonlocal mask branch
class NLMaskBranchDownUp(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(NLMaskBranchDownUp, self).__init__()
        
        MB_RB1 = []
        MB_RB1.append(NonLocalBlock2D(n_feat, n_feat // 2))
        MB_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        MB_Down = []
        MB_Down.append(nn.Conv2d(n_feat,n_feat, 3, stride=2, padding=1))
        
        MB_RB2 = []
        for i in range(2):
            MB_RB2.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
         
        MB_Up = []
        MB_Up.append(nn.ConvTranspose2d(n_feat,n_feat, 6, stride=2, padding=2))   
        
        MB_RB3 = []
        MB_RB3.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        MB_1x1conv = []
        MB_1x1conv.append(nn.Conv2d(n_feat,n_feat, 1, padding=0, bias=True))
        
        MB_sigmoid = []
        MB_sigmoid.append(nn.Sigmoid())

        self.MB_RB1 = nn.Sequential(*MB_RB1)
        self.MB_Down = nn.Sequential(*MB_Down)
        self.MB_RB2 = nn.Sequential(*MB_RB2)
        self.MB_Up  = nn.Sequential(*MB_Up)
        self.MB_RB3 = nn.Sequential(*MB_RB3)
        self.MB_1x1conv = nn.Sequential(*MB_1x1conv)
        self.MB_sigmoid = nn.Sequential(*MB_sigmoid)
    
    def forward(self, x):
        x_RB1 = self.MB_RB1(x)
        x_Down = self.MB_Down(x_RB1)
        x_RB2 = self.MB_RB2(x_Down)
        x_Up = self.MB_Up(x_RB2)
        x_preRB3 = x_RB1 + x_Up
        x_RB3 = self.MB_RB3(x_preRB3)
        x_1x1 = self.MB_1x1conv(x_RB3)
        mx = self.MB_sigmoid(x_1x1)

        return mx




## define residual attention module 
class ResAttModuleDownUpPlus(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(ResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(MaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB  = nn.Sequential(*RA_TB)
        self.RA_MB  = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx


## define nonlocal residual attention module 
class NLResAttModuleDownUpPlus(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(NLResAttModuleDownUpPlus, self).__init__()
        RA_RB1 = []
        RA_RB1.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_TB = []
        RA_TB.append(TrunkBranch(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_MB = []
        RA_MB.append(NLMaskBranchDownUp(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        RA_tail = []
        for i in range(2):
            RA_tail.append(ResBlock(conv, n_feat, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        
        self.RA_RB1 = nn.Sequential(*RA_RB1)
        self.RA_TB  = nn.Sequential(*RA_TB)
        self.RA_MB  = nn.Sequential(*RA_MB)
        self.RA_tail = nn.Sequential(*RA_tail)

    def forward(self, input):
        RA_RB1_x = self.RA_RB1(input)
        tx = self.RA_TB(RA_RB1_x)
        mx = self.RA_MB(RA_RB1_x)
        txmx = tx * mx
        hx = txmx + RA_RB1_x
        hx = self.RA_tail(hx)

        return hx


class _ResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_ResGroup, self).__init__()
        modules_body = []
        modules_body.append(ResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res

class _NLResGroup(nn.Module):
    def __init__(self, conv, n_feats, kernel_size, act, res_scale):
        super(_NLResGroup, self).__init__()
        modules_body = []
        modules_body.append(NLResAttModuleDownUpPlus(conv, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1))
        modules_body.append(conv(n_feats, n_feats, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return res

class RNAN(nn.Module):
    def __init__(self, opt):
        super(RNAN, self).__init__()
        
        n_resgroups = opt.n_resgroups
        n_feats = opt.n_feats
        kernel_size = 3
        reduction = opt.reduction
        act = nn.ReLU(True)


        print(n_resgroup2,n_resblock,n_feats,kernel_size,reduction,act)
        
        # RGB mean for DIV2K 1-800
        # rgb_mean = (0.4488, 0.4371, 0.4040)
        # rgb_std = (1.0, 1.0, 1.0)    
        # self.sub_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std)
        
        # define head module
        modules_head = [conv(opt.nch_in, n_feats, kernel_size)]
        
        # define body module
        modules_body_nl_low = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=1)]
        modules_body = [
            _ResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=1) \
            for _ in range(n_resgroups - 2)]
        modules_body_nl_high = [
            _NLResGroup(
                conv, n_feats, kernel_size, act=act, res_scale=1)]
        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module       
        modules_tail = [
            Upsampler(conv, opt.scale, n_feats, act=False),
            conv(n_feats, opt.nch_out, kernel_size)]

        # self.add_mean = MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*modules_head)
        self.body_nl_low = nn.Sequential(*modules_body_nl_low)
        self.body = nn.Sequential(*modules_body)
        self.body_nl_high = nn.Sequential(*modules_body_nl_high)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):

        # x = self.sub_mean(x)
        feats_shallow = self.head(x)

        res = self.body_nl_low(feats_shallow)
        res = self.body(res)
        res = self.body_nl_high(res)
        res += feats_shallow

        res_main = self.tail(res)

        # res_main = self.add_mean(res_main)

        return res_main  












class FourierNet(nn.Module):

    def __init__(self):
        super(FourierNet, self).__init__()
        self.inp = nn.Linear(85*85*9,85*85)
    

    def forward(self, x):
        x = x.view(-1,85*85*9)
        x = (self.inp(x))
#         x = (self.lay1(x))
        x = x.view(-1,1,85,85)
        return x


class FourierConvNet(nn.Module):

    def __init__(self):
        super(FourierConvNet, self).__init__()
        
        
        # self.inp = nn.Conv2d(18,32,3, stride=1, padding=1)
        # self.lay1 = nn.Conv2d(32,32,3, stride=1, padding=1)
        # self.lay2 = nn.Conv2d(32,32,3, stride=1, padding=1)
        # self.lay3 = nn.Conv2d(32,32,3, stride=1, padding=1)

        # self.pool = nn.MaxPool2d(2,2)
        # self.out = nn.Conv2d(32,1,3, stride=1, padding=1)

        # self.labels = nn.Linear(4096,18)
        
        self.inc = inconv(18, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, 9) # two channels for complex


    def forward(self, x):
        # x = self.inp(x)
        
        # x = torch.rfft(x,2,onesided=False)
        # # x = torch.log( torch.abs(x) + 1 )

        # x = x.permute(0,1,4,2,3) # put real and imag parts after stack index
        # x = x.contiguous().view(-1,18,256,256)

        # x = F.relu(self.inp(x))
        
        # x = self.pool(x) # to 128
        # x = F.relu(self.lay2(x))
        # x = self.pool(x) # to 64
        # x = F.relu(self.lay3(x))
        
        # x = self.out(x)

        # x = x.view(-1,4096)

        # x = self.labels(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)

        x = torch.log(torch.abs(x))

        # x = x.permute(0,2,3,1)
        # x = torch.irfft(x,2,onesided=False)
        return x


    #     super(UNet, self).__init__()
        # self.inc = inconv(n_channels, 64)
        # self.down1 = down(64, 128)
        # self.down2 = down(128, 256)
        # self.down3 = down(256, 512)
        # self.down4 = down(512, 512)
        # self.up1 = up(1024, 256)
        # self.up2 = up(512, 128)
        # self.up3 = up(256, 64)
        # self.up4 = up(128, 64)
        
        # if opt.task == 'segment':
        #     self.outc = outconv(64, 2)
        # else:
        #     self.outc = outconv(64, n_classes)

    #     # Initialize weights
    #     # self._init_weights()


    # def _init_weights(self):
    #     """Initializes weights using He et al. (2015)."""

    #     for m in self.modules():
    #         if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight.data)
    #             m.bias.data.zero_()


    # def forward(self, x):
    #     x1 = self.inc(x)
    #     x2 = self.down1(x1)
    #     x3 = self.down2(x2)
    #     x4 = self.down3(x3)
    #     x5 = self.down4(x4)
    #     x = self.up1(x5, x4)
    #     x = self.up2(x, x3)
    #     x = self.up3(x, x2)
    #     x = self.up4(x, x1)
    #     x = self.outc(x)
    #     return F.sigmoid(x)



class UNet_n2n(nn.Module):
    """Custom U-Net architecture for Noise2Noise (see Appendix, Table 2)."""

    def __init__(self, in_channels=3, out_channels=3, opt = {}):
        """Initializes U-Net."""

        super(UNet_n2n, self).__init__()

        # Layers: enc_conv0, enc_conv1, pool1
        self._block1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv(i), pool(i); i=2..5
        self._block2 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))

        # Layers: enc_conv6, upsample5
        self._block3 = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(48, 48, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self._block4 = nn.Sequential(
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        self._block5 = nn.Sequential(
            nn.Conv2d(144, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 96, 3, stride=2, padding=1, output_padding=1))
            #nn.Upsample(scale_factor=2, mode='nearest'))

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self._block6 = nn.Sequential(
            nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1))

        # Initialize weights
        self._init_weights()

        self.task = opt.task
        if opt.task == 'segment':
            self._block6 = nn.Sequential(
                nn.Conv2d(96 + in_channels, 64, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 2, 1))



    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        """Through encoder, then decoder by adding U-skip connections. """

        # Encoder
        pool1 = self._block1(x)
        pool2 = self._block2(pool1)
        pool3 = self._block2(pool2)
        pool4 = self._block2(pool3)
        pool5 = self._block2(pool4)

        # Decoder
        upsample5 = self._block3(pool5)
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self._block4(concat5)
        concat4 = torch.cat((upsample4, pool3), dim=1)
        upsample3 = self._block5(concat4)
        concat3 = torch.cat((upsample3, pool2), dim=1)
        upsample2 = self._block5(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1)
        upsample1 = self._block5(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)

        # Final activation
        return self._block6(concat1)



# ------------------ Alternative UNet implementation (batchnorm. outcommented)


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            # nn.Conv2d(in_ch,in_ch, 2, stride=2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes,opt):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        
        if opt.task == 'segment':
            self.outc = outconv(64, 2)
        else:
            self.outc = outconv(64, n_classes)

        # Initialize weights
        # self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)


class UNet60M(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet60M, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.down5 = down(1024, 1024)
        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 64)
        self.up5 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        x = self.outc(x)
        return F.sigmoid(x)


class UNetRep(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetRep, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 128)
        self.up1 = up1(256, 128, 128)
        self.up2 = up1(192, 64, 128)
        
        self.outc = outconv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)

        for _ in range(3):
            x2 = self.down1(x1)
            x3 = self.down2(x2)
            x = self.up1(x3,x2)
            x1 = self.up2(x,x1)

        # x6 = self.down5(x5)
        # x = self.up1(x6, x5)
        # x = self.up2(x, x4)
        # x = self.up3(x, x3)
        # x = self.up4(x, x2)
        # x = self.up5(x, x1)
        x = self.outc(x1)
        return F.sigmoid(x)




# ------------------- UNet Noise2noise implementation
 
class single_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv2, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class up1(nn.Module):
    def __init__(self, in_ch, out_ch, convtr, bilinear=False):
        super(up1, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(convtr, convtr, 3, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x        

class up2(nn.Module):
    def __init__(self, in_ch, in_ch2, out_ch,out_ch2,convtr, bilinear=False):
        super(up2, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(convtr, convtr, 3, stride=2)

        # self.conv = double_conv(in_ch, out_ch)
        self.conv = nn.Conv2d(in_ch + in_ch2, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch2, 3, padding=1)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.conv2(x)
        return x

class down2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down2, self).__init__()
        self.mpconv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.Conv2d(in_ch,in_ch, 2, stride=2),
            single_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UNetGreedy(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetGreedy, self).__init__()
        self.inc = inconv(n_channels, 144)
        self.down1 = down(144, 144)
        self.down2 = down2(144, 144)
        self.down3 = down2(144, 144)
        self.down4 = down2(144, 144)
        self.down5 = down2(144, 144)
        self.up1 = up1(288, 288,144)
        self.up2 = up1(432, 288,288)
        self.up3 = up1(432, 288,288)
        self.up4 = up1(432, 288,288)
        self.up5 = up2(288, n_channels, 64, 32,288)
        self.outc = outconv2(32, n_classes)

    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x0)
        x = self.outc(x)
        return F.sigmoid(x)        


class UNet2(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet2, self).__init__()
        self.inc = inconv(n_channels, 48)
        self.down1 = down(48, 48)
        self.down2 = down2(48, 48)
        self.down3 = down2(48, 48)
        self.down4 = down2(48, 48)
        self.down5 = down2(48, 48)
        self.up1 = up1(96, 96,48)
        self.up2 = up1(144, 96,96)
        self.up3 = up1(144, 96,96)
        self.up4 = up1(144, 96,96)
        self.up5 = up2(96, n_channels, 64, 32,96)
        self.outc = outconv2(32, n_classes)

    def forward(self, x0):
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x0)
        x = self.outc(x)
        return F.sigmoid(x)        


class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(64, 96, 3, padding=1)
        self.conv22 = nn.Conv2d(96, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(96, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv6 = nn.Conv2d(64, 32, 5)
        # self.conv3 = nn.Conv2d(24, 48, 3, padding=1)
        self.fc = nn.Sequential(
            nn.Linear(6*6*32, 100),
            nn.ReLU(),
            nn.Dropout2d(p=0.2),
            nn.Linear(100, 1),
            nn.ReLU()
        )
#         self.fc2 = nn.Linear(100,50)
#         self.fc3 = nn.Linear(50,20)
#         self.fc4 = nn.Linear(20,1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv12(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        # x = F.relu(self.conv22(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)
        x = F.relu(self.conv5(x))
        x = self.pool(x)
        x = F.relu(self.conv6(x))
        x = self.pool(x)
        x = x.view(-1, 6*6*32)
        x = self.fc(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return x        



# --------------------- FFDNet
from torch.autograd import Function, Variable

def concatenate_input_noise_map(input, noise_sigma):
	r"""Implements the first layer of FFDNet. This function returns a
	torch.autograd.Variable composed of the concatenation of the downsampled
	input image and the noise map. Each image of the batch of size CxHxW gets
	converted to an array of size 4*CxH/2xW/2. Each of the pixels of the
	non-overlapped 2x2 patches of the input image are placed in the new array
	along the first dimension.

	Args:
		input: batch containing CxHxW images
		noise_sigma: the value of the pixels of the CxH/2xW/2 noise map
	"""
	# noise_sigma is a list of length batch_size
	N, C, H, W = input.size()
	dtype = input.type()
	sca = 2
	sca2 = sca*sca
	Cout = sca2*C
	Hout = H//sca
	Wout = W//sca
	idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

	# Fill the downsampled image with zeros
	if 'cuda' in dtype:
		downsampledfeatures = torch.cuda.FloatTensor(N, Cout, Hout, Wout).fill_(0)
	else:
		downsampledfeatures = torch.FloatTensor(N, Cout, Hout, Wout).fill_(0)

	# Build the CxH/2xW/2 noise map
	noise_map = noise_sigma.view(N, 1, 1, 1).repeat(1, C, Hout, Wout)

	# Populate output
	for idx in range(sca2):
		downsampledfeatures[:, idx:Cout:sca2, :, :] = \
			input[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]

	# concatenate de-interleaved mosaic with noise map
	return torch.cat((noise_map, downsampledfeatures), 1)

class UpSampleFeaturesFunction(Function):
	r"""Extends PyTorch's modules by implementing a torch.autograd.Function.
	This class implements the forward and backward methods of the last layer
	of FFDNet. It basically performs the inverse of
	concatenate_input_noise_map(): it converts each of the images of a
	batch of size CxH/2xW/2 to images of size C/4xHxW
	"""
	@staticmethod
	def forward(ctx, input):
		N, Cin, Hin, Win = input.size()
		dtype = input.type()
		sca = 2
		sca2 = sca*sca
		Cout = Cin//sca2
		Hout = Hin*sca
		Wout = Win*sca
		idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

		assert (Cin%sca2 == 0), \
			'Invalid input dimensions: number of channels should be divisible by 4'

		result = torch.zeros((N, Cout, Hout, Wout)).type(dtype)
		for idx in range(sca2):
			result[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca] = \
				input[:, idx:Cin:sca2, :, :]

		return result

	@staticmethod
	def backward(ctx, grad_output):
		N, Cg_out, Hg_out, Wg_out = grad_output.size()
		dtype = grad_output.data.type()
		sca = 2
		sca2 = sca*sca
		Cg_in = sca2*Cg_out
		Hg_in = Hg_out//sca
		Wg_in = Wg_out//sca
		idxL = [[0, 0], [0, 1], [1, 0], [1, 1]]

		# Build output
		grad_input = torch.zeros((N, Cg_in, Hg_in, Wg_in)).type(dtype)
		# Populate output
		for idx in range(sca2):
			grad_input[:, idx:Cg_in:sca2, :, :] = \
				grad_output.data[:, :, idxL[idx][0]::sca, idxL[idx][1]::sca]

		return Variable(grad_input)

# Alias functions
upsamplefeatures = UpSampleFeaturesFunction.apply




class UpSampleFeatures(nn.Module):
	r"""Implements the last layer of FFDNet
	"""
	def __init__(self):
		super(UpSampleFeatures, self).__init__()
	def forward(self, x):
		return upsamplefeatures(x)

class IntermediateDnCNN(nn.Module):
	r"""Implements the middel part of the FFDNet architecture, which
	is basically a DnCNN net
	"""
	def __init__(self, input_features, middle_features, num_conv_layers):
		super(IntermediateDnCNN, self).__init__()
		self.kernel_size = 3
		self.padding = 1
		self.input_features = input_features
		self.num_conv_layers = num_conv_layers
		self.middle_features = middle_features
		if self.input_features == 5:
			self.output_features = 4 #Grayscale image
		elif self.input_features == 15:
			self.output_features = 12 #RGB image
		else:
			self.output_features = 3            
			# raise Exception('Invalid number of input features')


		layers = []
		layers.append(nn.Conv2d(in_channels=self.input_features,\
								out_channels=self.middle_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		layers.append(nn.ReLU(inplace=True))
		for _ in range(self.num_conv_layers-2):
			layers.append(nn.Conv2d(in_channels=self.middle_features,\
									out_channels=self.middle_features,\
									kernel_size=self.kernel_size,\
									padding=self.padding,\
									bias=False))
			# layers.append(nn.BatchNorm2d(self.middle_features))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Conv2d(in_channels=self.middle_features,\
								out_channels=self.output_features,\
								kernel_size=self.kernel_size,\
								padding=self.padding,\
								bias=False))
		self.itermediate_dncnn = nn.Sequential(*layers)
	def forward(self, x):
		out = self.itermediate_dncnn(x)
		return out

class FFDNet(nn.Module):
	r"""Implements the FFDNet architecture
	"""
	def __init__(self, num_input_channels, test_mode=False):
		super(FFDNet, self).__init__()
		self.num_input_channels = num_input_channels
		self.test_mode = test_mode
		if self.num_input_channels == 1:
			# Grayscale image
			self.num_feature_maps = 64
			self.num_conv_layers = 15
			self.downsampled_channels = 5
			self.output_features = 4
		elif self.num_input_channels == 3:
			# RGB image
			self.num_feature_maps = 96
			self.num_conv_layers = 12
			self.downsampled_channels = 15
			self.output_features = 12
		else:
			raise Exception('Invalid number of input features')

		self.intermediate_dncnn = IntermediateDnCNN(\
				input_features=self.downsampled_channels,\
				middle_features=self.num_feature_maps,\
				num_conv_layers=self.num_conv_layers)
		self.upsamplefeatures = UpSampleFeatures()

	def forward(self, x, noise_sigma):
		concat_noise_x = concatenate_input_noise_map(\
				x.data, noise_sigma.data)
		if self.test_mode:
			concat_noise_x = Variable(concat_noise_x, volatile=True)
		else:
			concat_noise_x = Variable(concat_noise_x)
		h_dncnn = self.intermediate_dncnn(concat_noise_x)
		pred_noise = self.upsamplefeatures(h_dncnn)
		return pred_noise


class DNCNN(nn.Module):
	r"""Implements the DNCNNNet architecture
	"""
	def __init__(self, num_input_channels, test_mode=False):
		super(DNCNN, self).__init__()
		self.num_input_channels = num_input_channels
		self.test_mode = test_mode
		if self.num_input_channels == 1:
			# Grayscale image
			self.num_feature_maps = 64
			self.num_conv_layers = 15
			self.downsampled_channels = 5
			self.output_features = 4
		elif self.num_input_channels == 3:
			# RGB image
			self.num_feature_maps = 96
			self.num_conv_layers = 12
			self.downsampled_channels = 15
			self.output_features = 12
		else:
			raise Exception('Invalid number of input features')

		self.intermediate_dncnn = IntermediateDnCNN(\
				input_features=self.num_input_channels,\
				middle_features=self.num_feature_maps,\
				num_conv_layers=self.num_conv_layers)

	def forward(self, x):
		dncnn = self.intermediate_dncnn(x)
		return dncnn





### ---------------------------- ESRGAN --------------------------------


class ESRGAN_FeatureExtractor(nn.Module):
    def __init__(self):
        super(ESRGAN_FeatureExtractor, self).__init__()
        vgg19_model = models.vgg19(pretrained=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):
        return self.vgg19_54(img)


class DenseResidualBlock(nn.Module):
    """
    The core module of paper: (Residual Dense Network for Image Super-Resolution, CVPR 18)
    """

    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        def block(in_features, non_linearity=True):
            layers = [nn.Conv2d(in_features, filters, 3, 1, 1, bias=True)]
            if non_linearity:
                layers += [nn.LeakyReLU()]
            return nn.Sequential(*layers)

        self.b1 = block(in_features=1 * filters)
        self.b2 = block(in_features=2 * filters)
        self.b3 = block(in_features=3 * filters)
        self.b4 = block(in_features=4 * filters)
        self.b5 = block(in_features=5 * filters, non_linearity=False)
        self.blocks = [self.b1, self.b2, self.b3, self.b4, self.b5]

    def forward(self, x):
        inputs = x
        for block in self.blocks:
            out = block(inputs)
            inputs = torch.cat([inputs, out], 1)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x


class GeneratorRRDB(nn.Module):
    def __init__(self, opt, filters=48, num_res_blocks=4, num_upsample=0):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(opt.nch_in, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)
        # Upsampling layers
        upsample_layers = []
        for _ in range(num_upsample):
            upsample_layers += [
                nn.Conv2d(filters, filters * 4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
            ]
        self.upsampling = nn.Sequential(*upsample_layers)
        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, opt.nch_out, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class ESRGAN_Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(ESRGAN_Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)