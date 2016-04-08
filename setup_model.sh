if [ ! -f VGG_ILSVRC_16_layers.caffemodel ]; then
    wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel
fi

python create_chainer_model.py
