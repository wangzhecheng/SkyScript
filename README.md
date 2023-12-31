# SkyScript

[[Paper]](https://arxiv.org/abs/2312.12856) [[Citing]](https://github.com/wangzhecheng/SkyScript) [Project Page] (under construction)

Welcome to the official repository of SkyScript. SkyScript is a large and semantically diverse image-text dataset for remote sensing images. It contains 5.2 million remote sensing image-text pairs in total, covering more than 29K distinct semantic tags. 

Remote sensing imagery, despite its broad applications in helping achieve Sustainable Development Goals and tackle climate change, has not fully benefited from the recent advancements of versatile, task-agnostic vision language models (VLMs). A key reason is that large and semantically diverse image-text datasets required for developing VLMs are not readily available for remote sensing images. Unlike natural images, remote sensing images and their associated text descriptions cannot be efficiently collected from the web at scale. Here we bridge this gap by connecting the “web” of geography—[OpenStreetMap (OSM)](https://www.openstreetmap.org) with multi-source, multi-resolution remote sensing images, resulting in SkyScript, a large-scale remote sensing image-text dataset with considerably rich semantic diversity.

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












