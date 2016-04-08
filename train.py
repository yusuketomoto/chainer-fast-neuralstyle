import numpy as np
import os
import argparse
from PIL import Image

from chainer import cuda, Variable, optimizers, serializers
from net import *

def gram_matrix(y):
    b, ch, w, h = y.data.shape
    features = F.reshape(y, (b, ch, w*h))
    gram = F.batch_matmul(features, features, transb=True)/np.float32(ch*w*h)
    return gram

parser = argparse.ArgumentParser(description='Real-time style transfer')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-d', default='dataset', type=str,
                    help='dataset directory path (according to the paper, use MSCOCO 80k images)')
parser.add_argument('--style_image', '-s', type=str, required=True,
                    help='style image path')
parser.add_argument('--batchsize', '-b', type=int, default=4,
                    help='batch size (default value is 4)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', default='out', type=str,
                    help='output model file path without extension')
parser.add_argument('--total_variation_regularization', '-v', default=10e-4, type=float,
                    help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
parser.add_argument('--lambda_feat', default=5e0, type=float)
parser.add_argument('--lambda_style', default=1e2, type=float)
parser.add_argument('--epoch', '-e', default=2, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--checkpoint', '-c', default=0, type=int)
args = parser.parse_args()

n_epoch = args.epoch
lambda_f = args.lambda_feat
lambda_s = args.lambda_style
fs = os.listdir(args.dataset)
imagepaths = []
for fn in fs:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.dataset,fn)
        imagepaths.append(imagepath)
n_data = len(imagepaths)
print 'num traning images:', n_data
n_iter = n_data / args.batchsize
print n_iter, 'iterations,', n_epoch, 'epochs'

model = FastStyleNet()
vgg = VGG()
serializers.load_npz('vgg16.model', vgg)
if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    vgg.to_gpu()
xp = np if args.gpu < 0 else cuda.cupy

O = optimizers.Adam(alpha=args.lr)
O.setup(model)

style = vgg.preprocess(np.asarray(Image.open(args.style_image).convert('RGB').resize((256,256)), dtype=np.float32))
style = xp.asarray(style, dtype=xp.float32)
style_b = xp.zeros((args.batchsize,) + style.shape, dtype=xp.float32)
for i in range(args.batchsize):
    style_b[i] = style
feature_s = vgg(Variable(style_b, volatile=True))
gram_s = [gram_matrix(y) for y in feature_s]

for epoch in range(n_epoch):
    print 'epoch', epoch
    for i in range(n_iter):
        model.zerograds()
        vgg.zerograds()

        indices = range(i * args.batchsize, (i+1) * args.batchsize)
        x = xp.zeros((args.batchsize, 3, 256, 256), dtype=xp.float32)
        for j in range(args.batchsize):
            x[j] = xp.asarray(Image.open(imagepaths[i*args.batchsize + j]).convert('RGB').resize((256,256)), dtype=np.float32).transpose(2, 0, 1)

        x -= 120 # subtract mean
        xc = Variable(x.copy(), volatile=True)
        x = Variable(x)

        y = model(x)

        feature = vgg(xc)
        feature_hat = vgg(y)

        L_feat = lambda_f * F.mean_squared_error(Variable(feature[2].data), feature_hat[2]) # compute for only the output of layer conv3_3

        L_style = Variable(xp.zeros((), dtype=np.float32))
        for f, f_hat, g_s in zip(feature, feature_hat, gram_s):
            L_style += lambda_s * F.mean_squared_error(gram_matrix(f_hat), Variable(g_s.data))

        L = L_feat + L_style

        print '(epoch {}) batch {}/{}... training loss is...{}'.format(epoch, i, n_iter, L.data)

        L.backward()
        O.update()

        if args.checkpoint > 0 and i % args.checkpoint == 0:
            serializers.save_npz('models/style_{}_{}.model'.format(epoch, i), model)

    print 'save "style.model"'
    serializers.save_npz('models/style_{}.model'.format(epoch), model)

serializers.save_npz('models/style.model'.format(epoch), model)
