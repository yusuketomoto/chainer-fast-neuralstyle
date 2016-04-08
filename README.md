# Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"

This repository is currently in progress.
Any improvements by sending pull-request are welcomed.

### Known issue
Some glitches occur on the edge of generated image.

## Requirement
- [Chainer](https://github.com/pfnet/chainer)
```
$ pip install chainer
```

## Prerequisite
Download VGG16 model and convert it into smaller file so that we use only the convolution layers which are 10% of the entire model.
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

## Difference from paper
- Use linear activation instead of scaled hyperbolic tangent on output layer.
- Convolution kernel size 4 instead of 3.
- Not implemented total variation regularization.
- Not sure whether adding/subtracting mean image is needed or not. In this implementation mean image subtraction is done before input image is fed into "image transformation network".

## License
MIT

## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](arxiv.org/abs/1603.08155)
- [chainer-gogh](https://github.com/mattya/chainer-gogh.git) chainer implementation of neural-style. I heavily referenced it, thanks to the author.
