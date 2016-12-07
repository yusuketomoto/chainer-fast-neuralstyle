from __future__ import print_function
from chainer import link
from chainer.links.caffe import CaffeFunction
from chainer import serializers
from net import *

# http://qiita.com/tabe2314/items/6c0c1b769e12ab1e2614
def copy_model(src, dst):
    assert isinstance(src, link.Chain)
    assert isinstance(dst, link.Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, link.Chain):
            copy_model(child, dst_child)
        if isinstance(child, link.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print('Ignore %s because of parameter mismatch' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = a[1].data
            print('Copy %s' % child.name)

print('load VGG16 caffemodel')
ref = CaffeFunction('VGG_ILSVRC_16_layers.caffemodel')
vgg = VGG()
print('copy weights')
copy_model(ref, vgg)

print('save "vgg16.model"')
serializers.save_npz('vgg16.model', vgg)
