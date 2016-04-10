# Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
Fast artistic style transfer by using feed forward network.

## Requirement
- [Chainer](https://github.com/pfnet/chainer)
```
$ pip install chainer
```

## Prerequisite
Download VGG16 model and convert it into smaller file so that we use only the convolutional layers which are 10% of the entire model.
```
sh setup_model.sh
```

## Train
Need to train one image transformation network model per one style target.
According to the paper, the models are trained on the [Microsoft COCO dataset](http://mscoco.org/dataset/#download).
```
python train.py -s <style_image_path> -d <training_dataset_path> -g 0
```

## Generate
```
python generate.py <input_image_path> -m <model_path> -o <output_image_path>
```

This repo has a pretrained model which was trained with "The starry night" by "Vincent van Gogh" as an example.
- example:
```
python generate.py sample_images/input_0.jpg -m models/starrynight.model -o sample_images/output_0.jpg
```

## Difference from paper
- Convolution kernel size 4 instead of 3.
- Currently impossible to train against mini-batches. Computation of Gram-matrix with "chainer.functions.batch_matmul" didn't work for me (causes some glitches on the edge of generated image.), so I use "chainer.functions.matmul" and train without mini-batches. Any improvements against this problem by sending pull-request are welcomed.
- Not sure whether adding/subtracting mean image is needed or not. In this implementation mean image subtraction is done before input image is fed into "image transformation network".

## License
MIT

## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155)

Codes written in this repository based on following nice works, thanks to the author.
- [chainer-gogh](https://github.com/mattya/chainer-gogh.git) Chainer implementation of neural-style. I heavily referenced it.
- [chainer-cifar10](https://github.com/mitmul/chainer-cifar10) Residual block implementation is referred.
