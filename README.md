# Chainer implementation of "Perceptual Losses for Real-Time Style Transfer and Super-Resolution"
Fast artistic style transfer by using feed forward network.

**checkout [resize-conv](https://github.com/yusuketomoto/chainer-fast-neuralstyle/tree/resize-conv) branch which provides better result.**

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/tubingen.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_1.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_1.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/style_2.png" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_2.jpg" height="200px">

- input image size: 1024x768
- process time(CPU): 17.78sec (Core i7-5930K)
- process time(GPU): 0.994sec (GPU TitanX)


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
python train.py -s <style_image_path> -d <training_dataset_path> -g <use_gpu ? gpu_id : -1>
```

## Generate
```
python generate.py <input_image_path> -m <model_path> -o <output_image_path> -g <use_gpu ? gpu_id : -1>
```

This repo has pretrained models as an example.

- example:
```
python generate.py sample_images/tubingen.jpg -m models/composition.model -o sample_images/output.jpg
```
or
```
python generate.py sample_images/tubingen.jpg -m models/seurat.model -o sample_images/output.jpg
```

#### Transfer only style but not color (**--keep_colors option**)
`python generate.py <input_image_path> -m <model_path> -o <output_image_path> -g <use_gpu ? gpu_id : -1> --keep_colors`

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_1.jpg" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_keep_colors_1.jpg" height="200px">

<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_2.jpg" height="200px">
<img src="https://raw.githubusercontent.com/yusuketomoto/chainer-fast-neuralstyle/master/sample_images/output_keep_colors_2.jpg" height="200px">


## A collection of pre-trained models
Fashizzle Dizzle created pre-trained models collection repository, [chainer-fast-neuralstyle-models](https://github.com/gafr/chainer-fast-neuralstyle-models). You can find a variety of models.

## Difference from paper
- Convolution kernel size 4 instead of 3.
- Training with batchsize(n>=2) causes unstable result.

## No Backward Compatibility
##### Jul. 19, 2016
This version is not compatible with the previous versions. You can't use models trained by the previous implementation. Sorry for the inconvenience!

## License
MIT

## Reference
- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155)

Codes written in this repository based on following nice works, thanks to the author.

- [chainer-gogh](https://github.com/mattya/chainer-gogh.git) Chainer implementation of neural-style. I heavily referenced it.
- [chainer-cifar10](https://github.com/mitmul/chainer-cifar10) Residual block implementation is referred.
