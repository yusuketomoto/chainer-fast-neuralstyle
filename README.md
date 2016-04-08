# Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"

### Note:
This repository is currently in progress.

## Requirement
- [Chainer](https://github.com/pfnet/chainer)
```
$ pip install chainer
```

## Prerequisite
Download VGG16 model and convert it into smaller file so that we use only the convolution layers which are 10% of entire model.
```
sh setup_model.sh
```

## Train
Need to train one image transformation network model per one style target.
According to the paper, the models are trained on the [Microsoft COCO dataset](http://mscoco.org/dataset/#download).
```
python train.py -s <style_image_path> -d <training_dataset_path> -g 0
```

## Generate your own image
```
python generate.py <input_image_path> -m <model_path> -o <output_image_path>
```

## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](arxiv.org/abs/1603.08155)
- [chainer-gogh](https://github.com/mattya/chainer-gogh.git) chainer implementation of neural-style. I heavily referenced it, super helpful.
