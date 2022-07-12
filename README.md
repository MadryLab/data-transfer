# A Data-Based Perspective on Transfer Learning.

This repository contains the code of our paper:

**A Data-Based Perspective on Transfer Learning** </br>
*Saachi Jain\*, Hadi Salman\*, Alaa Khaddaj\*, Eric Wong, Sung Min Park, Aleksander Madry*  <br>
[Paper](TODO) - [Blog post](http://gradientscience.org/data-transfer/)


```bibtex
@InProceedings{jain2022transfer,
    title={A Data-Based Perspective on Transfer Learning},
    author={Saachi Jain and Hadi Salman and Alaa Khaddaj and Eric Wong and Sung Min Park and Aleksander Madry},
    year={2022},
    booktitle={ArXiv preprint arXiv:XXXXX}
}
```

The major content of our repo are:

* [src/](src): Contains all our code for running full transfer pipeline.
* [configs/](configs): Contains the config files that training codes expect. These config files contain the hyperparams for each transfer tasks.
* [analysis/](analysis): Contains code for all the analysis we do in our paper.

## Getting started
*Our code relies on the [FFCV Library](https://ffcv.io/). To install this library along with other dependencies including PyTorch, follow the instructions below.*

```
conda create -n ffcv python=3.9 cupy pkg-config compilers libjpeg-turbo opencv pytorch torchvision cudatoolkit=11.3 numba -c pytorch -c conda-forge 
conda activate ffcv
pip install ffcv
```

## Full pipeline: Train source model and transfer to various downstream tasks

To train an ImageNet model and transfer it to all the datasets we consider in the paper, simply run:

```
python src/train_imagenet_class_subset.py \
                        --config-file configs/base_config.yaml \
                        --training.data_root $PATH_TO_DATASETS \
                        --out.output_pkl_dir $OUTDIR

```
where `$OUTDIR` is the output directory of your choice, and `$PATH_TO_DATASETS` is the path where the datasets exists (see below).

The config file `configs/base_config.yaml` contains all the hyperparameters needed for this experiment. For example, you can specify which downstream tasks you want to transfer to, or how many Imagenet class to train on the source model.

## Calculating influences
Use `analysis/data_compressors/2_20_compressor.py` to compress model results into a summary file. Then use `analysis/compute_influences.py` to compute the influences. In a notebook, simply run the following code:

```python
sf = <SUMMARY FILE FOLDER>
ds = compute_influences.SummaryFileDataSet(sf, dataset, INFLUENCE_KEY, keyword)
dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=False, drop_last=False)
infl = compute_influences.batch_calculate_influence(dl, len(val_labels), 1000, div=True)
```

## Running counterfactual experiment
Once influences have been computed, we can now run counterfactual experiments by removing top or bottom influencing classes and run transfer learning again. This can be done by running:
```
python src/counterfactuals_main.py\
            --config-file configs/base_config.yaml\
            --training.transfer_task ${TASK}\
            --out.output_pkl_dir ${OUT_DIR}\
            --counterfactual.cf_target_dataset ${DATASET}\
            --counterfactual.cf_infl_order_file ${INFL_ORDER_FILE} \
            --data.num_classes -1 \
            --counterfactual.cf_order TOP \
            --counterfactual.cf_num_classes_min ${MIN_STEPS} \
            --counterfactual.cf_num_classes_max ${MAX_STEPS} \
            --counterfactual.cf_num_classes_step ${STEP_SIZE} \
            --counterfactual.cf_type CLASS
```

## Datasets that we use (see our paper for citations) 
* aircraft ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/fgvc-aircraft-2013b.tar.gz?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* birds ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/birdsnap.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* caltech101 ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/caltech101.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* caltech256 ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/caltech256.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* cifar10 **(Automatically downloaded when you run the code)**
* cifar100 **(Automatically downloaded when you run the code)**

* flowers ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/flowers.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* food ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/food.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* pets ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/pets.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* stanford_cars ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/stanford_cars.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))
* SUN397 ([Download]( https://robustnessws4285631339.blob.core.windows.net/public-datasets/SUN397.tar?sv=2020-08-04&ss=bfqt&srt=sco&sp=rwdlacupitfx&se=2051-10-06T07:09:59Z&st=2021-10-05T23:09:59Z&spr=https,http&sig=U69sEOSMlliobiw8OgiZpLTaYyOA5yt5pHHH5%2FKUYgI%3D
))

We have created an [FFCV](https://ffcv.io/) version of each of these datasets to enable super fast training. We will make these datasets available soon!

## Download our data
Coming soon!

## Download our pretrained models
Coming soon!

## A detailed demo
Coming soon!
