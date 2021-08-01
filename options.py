import argparse

# training options
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='edsr', help='model to use')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--norm', type=str, default='', help='if normalization should not be used')
parser.add_argument('--nepoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--saveinterval', type=int, default=10, help='number of epochs between saves')
parser.add_argument('--ntrain', type=int, default=0, help='number of samples to train on')
parser.add_argument('--scheduler', type=str, default='', help='options for a scheduler, format: stepsize,gamma')
parser.add_argument('--log', action='store_true')

# data
parser.add_argument('--dataset', type=str, default='fouriersim', help='dataset to train')
parser.add_argument('--imageSize', type=int, default=24, help='the low resolution image size')
parser.add_argument('--weights', type=str, default='', help='model to retrain from')
parser.add_argument('--basedir', type=str, default='', help='path to prepend to all others paths: root, output, weights')
parser.add_argument('--root', type=str, default='/auto/homes/cnc39/phd/datasets', help='dataset to train')
parser.add_argument('--out', type=str, default='results', help='folder to output model training results')
parser.add_argument('--disposableTrainingData', action='store_true', help='whether to delete training data after training')

# computation
parser.add_argument('--workers', type=int, default=6, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')

# restoration options
parser.add_argument('--task', type=str, default='sr', help='restoration task')
parser.add_argument('--scale', type=int, default=4, help='low to high resolution scaling factor')
parser.add_argument('--nch_in', type=int, default=3, help='colour channels in input')
parser.add_argument('--nch_out', type=int, default=3, help='colour channels in output')

# architecture options
parser.add_argument('--narch', type=int, default=0, help='architecture-dependent parameter')
parser.add_argument('--n_resblocks', type=int, default=10, help='number of residual blocks')
parser.add_argument('--n_resgroups', type=int, default=10, help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16, help='number of feature maps reduction')
parser.add_argument('--n_feats', type=int, default=64, help='number of feature maps')

# test options
parser.add_argument('--ntest', type=int, default=10, help='number of images to test per epoch or test run')
parser.add_argument('--testinterval', type=int, default=1, help='number of epochs between tests during training')
parser.add_argument('--test', action='store_true')
parser.add_argument('--cpu', action='store_true') # not supported for training
parser.add_argument('--batchSize_test', type=int, default=1, help='input batch size for test loader')
parser.add_argument('--plotinterval', type=int, default=1, help='number of epochs between plotting')
parser.add_argument('--nplot', type=int, default=4, help='number of plots in a test')

