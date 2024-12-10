# SkyScript

[[Paper]](https://arxiv.org/abs/2312.12856) [[Citing]](https://github.com/wangzhecheng/SkyScript?tab=readme-ov-file#citing) [Project Page] (under construction)

Welcome to the official repository of SkyScript. SkyScript is a large and semantically diverse image-text dataset for remote sensing images. It contains 5.2 million remote sensing image-text pairs in total, covering more than 29K distinct semantic tags. The paper is accepted by and to appear in AAAI 2024.

Remote sensing imagery, despite its broad applications in helping achieve Sustainable Development Goals and tackle climate change, has not fully benefited from the recent advancements of versatile, task-agnostic vision language models (VLMs). A key reason is that large and semantically diverse image-text datasets required for developing VLMs are not readily available for remote sensing images. Unlike natural images, remote sensing images and their associated text descriptions cannot be efficiently collected from the web at scale. Here we bridge this gap by connecting the “web” of geography—[OpenStreetMap (OSM)](https://www.openstreetmap.org) with multi-source, multi-resolution remote sensing images, resulting in SkyScript, a large-scale remote sensing image-text dataset with considerably rich semantic diversity.

The model implementation of CLIP is adapted from the [OpenCLIP](https://github.com/mlfoundations/open_clip) repository. 

**News [2024/12]** We have converted raw OSM tags into free-form, smoother, and more natural language description using ChatGPT for each image. These new datasets (CSV files) can be downloaded with the URL links [Full 5M dataset](https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/dataframe/SkyScript_train_unfiltered_5M_language_polished.csv), [Top 50%](https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/dataframe/SkyScript_train_top50pct_filtered_by_CLIP_laion_RS_language_polished.csv), [Top 30%](https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/dataframe/SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv).

## Download

### Download SkyScript dataset

#### Download image files
The full dataset contains 5.2 million image-text pairs. They are partitioned into 6 parts. The image files can be downloaded using the following command:

```
curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/images2.zip
```

**Please replace `images2.zip` with filenames ranging from `images2.zip` to `images7.zip` to download all 6 parts**. Each zip file can be uncompressed into a folder. Image files (.jpg) are contained in these folders. The filename of each image file contains key meta information (object ID, image source, year) about the image. For example, ``a198234555_CH_19.jpg`` means: 

1. The image is retrieved with the OSM object 198234555 as the focus object (not necessarily at the image center).
2. The image source is `CH`, which is the alias of `ORTHO/Switzerland/SWISSIMAGE/10cm` (SWISSIMAGE 10cm RGB imagery from [Google Earth Engine](https://earthengine.google.com/)). The conversion between the image source alias code and Google Earth Engine image collection name is in the file .
3. The image was captured in 2019.

#### Download meta files
In addition to the meta information contained in the image file name, there is a meta file (.pickle) containing other information corresponding to each image. These meta files can be downloaded using the following command:

```
curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/meta2.zip
```

**Please replace `meta2.zip` with filenames ranging from `meta2.zip` to `meta7.zip` to download all 6 parts**. Each zip file can be uncompressed into a folder. Each meta file (.pickle) has the same name as its corresponding image file (except the file extension). Each meta file can be loaded with `pickle` library as a Python dictionary, which contains the following information:

1. `box`: the bounding box (latitute/longitude) of each image, represented as a tuple of four numerical values `(westernmost_longitude, southernmost_latitude, easternmost_longitude, northernmost_latitude)`.
2. `time`: image acquisition time, which is extracted from the Google Earth Engine. It is represented as a tuple containing five elements `(year, month, day, hour, minute)`.
3. `center_tags`: a dictionary containing the visually groundable semantic tags (key-value pairs) of the focus object. The focus object is used for locating and retrieving the image but not necessarily at the image center.
4. `surrounding_tags`: a list of dictionaries. Each dictionary contains the visually groundable semantic tags (key-value pairs) of one of the surrounding objects in the image.

#### Download captions

Image-text pairs are represented as a CSV file containing (1) a column called `filepath` (the relative path to the image file); (2) a column called `title` (the caption describing the focus object only); (3) a column called `title_multi_objects` (the caption describing both the focus object and surrounding objects.

These CSV files can be downloaded with the following command:

```
curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/dataframe/{FILENAME}
```

Here the `{FILENAME}` should be replaced with the following:

* `SkyScript_train_unfiltered_5M.csv`: all image-text pairs for **training** without any filtering. It contains 5.1 million image-text pairs.

* `SkyScript_train_top30pct_filtered_by_CLIP_openai.csv`: The top 30% most similar image-text pairs for **training**. The similarity is determined by the original CLIP model (OpenAI's checkpoint). It contains 1.5 million image-text pairs.
* `SkyScript_train_top50pct_filtered_by_CLIP_openai.csv`: The top 50% most similar image-text pairs for **training**. The similarity is determined by the original CLIP model (OpenAI's checkpoint). It contains 2.6 million image-text pairs.
* `SkyScript_val_5K_filtered_by_CLIP_openai.csv`: **Validation set** used during training. The similarity is determined by the original CLIP model (OpenAI's checkpoint). It contains 5K image-text pairs.
* `SkyScript_test_30K_filtered_by_CLIP_openai.csv`: **Test set** used for testing cross-modal retrieval performance. The similarity is determined by the original CLIP model (OpenAI's checkpoint). It contains 30K image-text pairs.
  
* `SkyScript_train_top30pct_filtered_by_CLIP_laion_RS.csv`: The top 30% most similar image-text pairs for **training**. The similarity is determined by the CLIP-laion-RS model. It contains 1.5 million image-text pairs.
* `SkyScript_train_top50pct_filtered_by_CLIP_laion_RS.csv`: The top 50% most similar image-text pairs for **training**. The similarity is determined by the CLIP-laion-RS model. It contains 2.6 million image-text pairs.
* `SkyScript_val_5K_filtered_by_CLIP_laion_RS.csv`: **Validation set** used during training. The similarity is determined by the CLIP-laion-RS model. It contains 5K image-text pairs.
* `SkyScript_test_30K_filtered_by_CLIP_laion_RS.csv`: **Test set** used for testing cross-modal retrieval performance. The similarity is determined by the CLIP-laion-RS model. It contains 30K image-text pairs.
* `SkyScript_train_unfiltered_5M.csv`: **(New!)** Used ChatGPT to convert raw OSM tags into free-formed, smoother, and more natural language descripion. It contains 4.7 million image-text pairs.
* `SkyScript_train_top30pct_filtered_by_CLIP_laion_RS_language_polished.csv`: **(New!)** Used ChatGPT to convert raw OSM tags into free-formed, smoother, and more natural language descripion. The top 30% most similar image-text pairs (1.5 million). The similarity is determined by the CLIP-laion-RS model.
* `SkyScript_train_top50pct_filtered_by_CLIP_laion_RS_language_polished.csv`: **(New!)** Used ChatGPT to convert raw OSM tags into free-formed, smoother, and more natural language descripion. The top 50% most similar image-text pairs (2.6 million). The similarity is determined by the CLIP-laion-RS model.


**NOTE: Here the captions are automatically assembled from semantic tags using a rule-based approach, as described in the paper. We welcome and encourage researchers to use more advanced techniques (e.g., Large Language Model) to generate more natural and diverse captions from semantic tags (semantic tags are provided in the meta files).**


### Download benchmark datasets

The benchmark datasets can be downloaded using the following command:

```
curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/benchmark/{FILENAME}
```

Here the `{FILENAME}` should be replaced with the following:

* **Scene classification**: `{FILENAME}` should be replaced with `aid.zip` (AID), `eurosat.zip` (EuroSAT), `fmow.zip` (fMoW), `millionaid.zip` (MillionAID), `nwpu.zip` (NWPU-RESISC45), `patternnet.zip` (PatterNet), `rsicb256.zip` (RSI-CB256), and `SkyScript_cls.zip`. Here `SkyScript_cls.zip` is the in-domain test set (containing 70 classes) while the remaining ones are out-of-domain test sets (datasets not used for training).
* **Fine-grained classification**: `{FILENAME}` should be replaced with `roof_shape.zip` (roof shape classification), `smoothness.zip` (road smoothness classification), and `surface.zip` (road surface material classification).
* **Cross-modal retrieval**: `{FILENAME}` should be replaced with `RSICD.zip`, `RSITMD.zip`, and `ucmcaptions.zip` (UCM-Captions).

### Download model checkpoints

The model checkpoints can be downloaded using the following command:

```
curl -O https://opendatasharing.s3.us-west-2.amazonaws.com/SkyScript/ckpt/{MODEL_NAME}
```

Here the `{MODEL_NAME}` should be replaced with the following:

* `SkyCLIP_ViT_L14_top30pct.zip`: CLIP ViT-L14 model. Continual pretraining with `SkyScript_train_top30pct_filtered_by_CLIP_openai.csv` (using the captions from the `title` column).
* `SkyCLIP_ViT_L14_top50pct.zip`: CLIP ViT-L14 model. Continual pretraining with `SkyScript_train_top50pct_filtered_by_CLIP_openai.csv` (using the captions from the `title` column).
* `SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS.zip`: CLIP ViT-L14 model. Continual pretraining with `SkyScript_train_top30pct_filtered_by_CLIP_laion_RS.csv` (using the captions from the `title` column).
* `SkyCLIP_ViT_L14_top30pct_multi_objects.zip`: CLIP ViT-L14 model. Continual pretraining with `SkyScript_train_top30pct_filtered_by_CLIP_openai.csv` (using the captions from the `title_multi_objects` column).
* `SkyCLIP_ViT_B32_top50pct.zip`: CLIP ViT-B32 model. Continual pretraining with `SkyScript_train_top50pct_filtered_by_CLIP_openai.csv` (using the captions from the `title` column).
* `CLIP_ViT_L14_LAION_RS.zip`: CLIP ViT-L14 model. Continual pretraining with the remote sensing image subset of the [LAION-2B](https://huggingface.co/datasets/laion/laion2B-en) dataset.

## Testing

**Please specify the root directory of benchmark datasets (`BENCHMARK_DATASET_ROOT_DIR`) in [`benchmark_dataset_info.py`](https://github.com/wangzhecheng/SkyScript/blob/main/benchmark_dataset_info.py)**

```
BENCHMARK_DATASET_ROOT_DIR = '/PATH/TO/THE/ROOT/DIRECTORY/OF/BENCHMARK/DATASETS'
```

### Testing zero-shot classification performance (scene classification and fine-grained classification)

Please follow the [`test_scene_and_fine_grained_classification.ipynb`](https://github.com/wangzhecheng/SkyScript/blob/main/test_scene_and_fine_grained_classification.ipynb). Note: Please specify the local paths to benchmark datasets and model checkpoints in this notebook.

For reference, the average top-1 zero-shot scene classification performances (average of `aid`, `eurosat`, `fmow`, `millionaid`, `patternnet`, `nwpu`, and `rsicb`) for different model checkpoints are listed as below (run on a single NVIDIA A100 GPU):

| Model checkpoint name | Model type | Avg. top-1 accuracy (%) |
| ---      | ---      | ---      |
| laion2b_e16 (OpenCLIP) | CLIP ViT-B32 | 49.66 |
| SkyCLIP_ViT_B32_top50pct | CLIP ViT-B32 | 53.02 |
| OpenAI | CLIP ViT-L14 | 53.76 |
| CLIP_ViT_L14_LAION_RS | CLIP ViT-L14 | 57.87 |
| SkyCLIP_ViT_L14_top30pct | CLIP ViT-L14 | 59.91 |
| SkyCLIP_ViT_L14_top30pct_filtered_by_CLIP_laion_RS | CLIP ViT-L14 | 60.69 |
| SkyCLIP_ViT_B32_top50pct | CLIP ViT-L14 | 59.93 |


### Testing zero-shot cross-modal retrieval (image-to-text and text-to-image)

Please follow the [`cross_modal_retrieval.ipynb`](https://github.com/wangzhecheng/SkyScript/blob/main/cross_modal_retrieval.ipynb). Note: Please specify the local paths to benchmark datasets and model checkpoints in this notebook.

## Training

Multi-GPU training can be performed using the following command:

```
torchrun --nproc_per_node $NUM_GPUS customized_train_and_test.py \
    --root-data-dir=$ROOT_DATA_DIR \
    --name=$MODEL_NAME \
    --save-frequency 1 \
    --val-frequency 1 \
    --train-data=$TRAINING_CSV_PATH  \
    --val-data=$VAL_CSV_PATH  \
    --csv-img-key filepath \
    --csv-separator=',' \
    --csv-caption-key $CAPTION_KEY \
    --random-rotation \
    --warmup 1000 \
    --batch-size=$BATCH_SIZE \
    --lr=$LR \
    --wd=$WD \
    --epochs=20 \
    --workers=8 \
    --model=$MODEL \
    --pretrained=$PRETRAINED \
    --aug-cfg use_timm=True color_jitter=0.4 scale="(0.67, 1.0)" ratio="(0.5, 2.0)"
```

Here `$ROOT_DATA_DIR` is the root directory to the SkyScript dataset (e.g., if `$ROOT_DATA_DIR = '/home/ubuntu/data/SkyScript'`, then the full path to an image file is `/home/ubuntu/data/SkyScript/images2/....jpg`). `$NUM_GPUS` is the number of GPUs used for training (we used 4). `$MODEL_NAME` is a given name (string) of the model. `$TRAINING_CSV_PATH` is the local path to the training CSV file (e.g., `SkyScript_train_top50pct_filtered_by_CLIP_openai.csv`). `$VAL_CSV_PATH` is the local path to the validation CSV file (e.g., `SkyScript_val_5K_filtered_by_CLIP_openai.csv`). `$CAPTION_KEY` is the column name of captions (`title` or `title_multi_objects`). `$BATCH_SIZE` is the batch size (we used 128). `$LR` is the learning rate (we used 3e-9 for CLIP ViT-L14 and 1e-9 for CLIP ViT-B32). `$WD` is the weight decay (we used 1.0). `$MODEL` is the model type (we used `ViT-L-14` or `ViT-B-32`). `$PRETRAINED` is the pretrained checkpoint for model initialization (we used `openai` for `ViT-L-14` and `laion2b_e16` for `ViT-B-32`).

## Citing

If you found this dataset useful, please consider citing:

```bibtex
@article{wang2023skyscript,
  title={SkyScript: A Large and Semantically Diverse Vision-Language Dataset for Remote Sensing},
  author={Wang, Zhecheng and Prabha, Rajanie and Huang, Tianyuan and Wu, Jiajun and Rajagopal, Ram},
  journal={arXiv preprint arXiv:2312.12856},
  year={2023}
}
```















