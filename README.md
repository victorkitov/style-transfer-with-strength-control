# Style transfer with strength control
This is a realization of [Real-Time Style Transfer With Strength Control (Kitov, 2019)](https://arxiv.org/abs/1904.08643)  which lets training a single transformer network for stylization with different stylization strength.

# Functionality
Train transformer network once (for each style). Then apply stylizations by passing images with the content through the transformer. Specification of stylization strength allows control of stylization impact at inference time.

<div align='center'>
<img src="https://github.com/Apogentus/style-transfer-with-strength-control/blob/master/images/results/man_on_the_boat.jpg_la_muse.pth_0.1.jpg?raw=trueg" width="24%" />
<img src="https://github.com/Apogentus/style-transfer-with-strength-control/blob/master/images/results/man_on_the_boat.jpg_la_muse.pth_0.3.jpg?raw=trueg" width="24%" />
<img src="https://github.com/Apogentus/style-transfer-with-strength-control/blob/master/images/results/man_on_the_boat.jpg_la_muse.pth_1.0.jpg?raw=trueg" width="24%" />
<img src="https://github.com/Apogentus/style-transfer-with-strength-control/blob/master/images/results/man_on_the_boat.jpg_la_muse.pth_10.0.jpg?raw=trueg" width="24%" />
</div>


<div align='center'>
<img src=https://github.com/Apogentus/style-transfer-with-strength-control/blob/master/images/results/woman_telephone.jpg_feathers.pth_0.1.jpg?raw=true" width="24%" />
<img src="https://github.com/Apogentus/style-transfer-with-strength-control/blob/master/images/results/woman_telephone.jpg_feathers.pth_0.3.jpg?raw=true" width="24%" />
<img src="https://github.com/Apogentus/style-transfer-with-strength-control/blob/master/images/results/woman_telephone.jpg_feathers.pth_1.0.jpg?raw=true" width="24%" />
<img src="https://github.com/Apogentus/style-transfer-with-strength-control/blob/master/images/results/woman_telephone.jpg_feathers.pth_10.0.jpg?raw=true" width="24%" />
</div>

Supported image file formats are: __jpg__, __jpeg__, __png__.

# Install
Needed Python packages can be installed using [`conda`](https://www.anaconda.com/download/) package manager by running `conda env create -f environment.yaml`

# Stylization
`python test.py`

optional arguments:
*  `-h, --help`            show help message and exit
*  `--content CONTENT [CONTENT ...]`
                        sequence of content images to be stylized (default:
                        images/contents/bus.jpg)
*  `--out_dir OUT_DIR`     directory where stylized images will be stored
                        (default: images/results/)
*  `--model MODEL`         path to serialized model, obtained via train.py
                        (default: models/la_muse.pth)
*  `--style_strength STYLE_STRENGTH`
                        non-negative float parameter, controlling stylization
                        strength (default: 1)
*  `--use_parallel_gpu USE_PARALLEL_GPU`
                        model trained using single GPU or using
                        parallelization over multiple GPUs (default: False)
*  `--gpu_id GPU_ID`       GPU to use (defaut: 0)
*  `--scale_content SCALE_CONTENT`
                        scaling factor for content images (default:None, no
                        scaling)

# Training
`python train.py`

optional arguments:
*  `-h, --help`           show help message and exit
*  `--max_train_count MAX_TRAIN_COUNT`
                        training will stop after passing this number of images
                        (default: 160000)
*  `--log_batches_interval LOG_BATCHES_INTERVAL`
                        number of batches after which the training loss is
                        logged (default: 80)
*  `--style_image STYLE_IMAGE`
                        path to style-image (default:
                        images/styles/la_muse.jpg)
*  `--gpu_id GPU_ID`       GPU to use (default: 0)
*  `--style_weight STYLE_WEIGHT`
                        weighting factor for style loss (default: 100000)
*  `--tv_weight TV_WEIGHT`
                        weighting factor for total variation loss (default:
                        1e-05)
*  `--max_style_strength MAX_STYLE_STRENGTH`
                        during training style_strength will be sampled
                        randomly from
                        [0,style_strength_step,...max_style_strength]
                        (default: 10)
*  `--style_strength_step STYLE_STRENGTH_STEP`
                        during training style_strength will be sampled
                        randomly from
                        [0,style_strength_step,...max_style_strength]
                        (default: 0.1)
*  `--dataset DATASET`     path to content images dataset on which model will be
                        trained, should point to a folder, containing another
                        folder with images (default:
                        ../../Datasets/Contents/MS-COCO/train2014)
*  `--checkpoint_batches_interval CHECKPOINT_BATCHES_INTERVAL`
                        number of batches after which a checkpoint of the
                        trained model will be created (default: None)
*  `--max_style_pixels MAX_STYLE_PIXELS`
                        max size in total pixels count of style-image during
                        training, None for no scaling (default: 160000)
*  `--use_parallel_gpu USE_PARALLEL_GPU`
                        model trained using single GPU or using
                        parallelization over multiple GPUs (default: False)
*  `--image_size IMAGE_SIZE`
                        during training content images are resized to this
                        size along X and Y axis (default: 256)
*  `--batch_size BATCH_SIZE`
                        size of batches during training (default: 12)
*  `--lr LR`               learning rate (default: 0.001)
*  `--init_model INIT_MODEL`
                        path to model if need model finetuning (default: )
*  `--save_model_dir SAVE_MODEL_DIR`
                        path where model will be saved (default: models/)
*  `--checkpoint_model_dir CHECKPOINT_MODEL_DIR`
                        path to folder where checkpoints of trained models
                        will be saved (default: intermediate_models/)
*  `--seed SEED`           random seed (default: 1)
*  `--loss_averaging_window LOSS_AVERAGING_WINDOW`
                        window averaging for losses (this average is displayed
                        during training) (default: 500)
