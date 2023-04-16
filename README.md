# PAC-MAN

This repository contains the PyTorch implementation for the IEEE Access paper: [PAC-MAN: Multi-Relation Network in Social Community for Personalized Hashtag Recommendation](https://ieeexplore.ieee.org/document/9984162)
> Padungkiatwattana, U., & Maneeroj, S. (2022). PAC-MAN: Multi-Relation Network in Social Community for Personalized Hashtag Recommendation. IEEE Access, 10, 131202-131228.

-----------------------------------------------------------

### Contents
* [:rainbow: Introduction](#rainbow-introduction)
* [:book: Dependencies](#book-dependencies)
* [:octocat: Repository](#octocat-repository)
* [:gear: Configuration](#gear-configuration)
* [:chart_with_upwards_trend: Dataset](#chart_with_upwards_trend-dataset)
* [:rocket: Model Training](#rocket-model-training)
* [:pushpin: Checkpoint and Logging](#pushpin-checkpoint-and-logging)
* [:star: Citation](#star-citation)

-----------------------------------------------------------

## :rainbow: Introduction

<p align="left">
<img src="assets/PAC-MAN.png">
</p>

PAC-MAN is a novel integral model for personalized hashtag recommendation, which has three main contributions:

:bulb: *First*, to derive fruitful user and hashtag representation from higher-order multiple relations, we propose ***Multi-relational Attentive Network (MAN)*** by applying GNN to jointly capture relations in three communities: (1) user-hashtag interaction (e.g., post, retweet, like); (2) user-user social (e.g., follow); and (3) hashtag-hashtag co-occurrence.

:bulb: *Second*, to personalize content at the word level, ***Person-And-Content based BERT (PAC)*** extends BERT to input not only word representations from the microblog but also the fruitful user representation from MAN, allowing each word to be fused with user aspects.

:bulb: *Third*, to capture sequenceless hashtag correlations, the fruitful hashtag representations from MAN that contain the hashtag’s community perspectives are inserted into BERT to integrate with the hashtag’s word-semantic perspectives, and a hashtag prediction task is then conducted under the *mask concept* for the recommendation.

-----------------------------------------------------------

## :book: Dependencies
The script has been tested under the following dependencies:
* `torch==2.0.0`
* `transformers==4.27.4`
* `tensorboard==2.12.1`
* `numpy==1.24.2`
* `scipy==1.10.1`
* `omegaconf==2.3.0`
* `tqdm==4.65.0`  

Install all dependencies:
```bash
pip install -r requirements.txt
```

-----------------------------------------------------------

## :octocat: Repository
* `mangnn/` contains code for Multi-relational Attentive Network (MANGNN).
* `pacbert/` contains code for Person-And-Content based BERT (PACBERT). 

-----------------------------------------------------------

## :gear: Configuration
Manage configuration for the model at:  
* **MANGNN**: `mangnn/config/config.yaml`.  
* **PACBERT**: `pacbert/config/config.yaml`.

-----------------------------------------------------------

## :chart_with_upwards_trend: Dataset

* **MANGNN**:  
    Prepare datasets and organize them as follows: 
    ```
    data
    └─ twitter
        └─ twitter_train.npy
        └─ twitter_val.npy
        └─ twitter_test.npy
        └─ networks
            └─ networks.json
            └─ follow.npy
            └─ post.npy
            └─ like.npy
            └─ retweet.npy
            └─ cooccur.npy
    ```
    
    <details>
    <summary>Here are the details of the dataset:</summary>

    * `twitter_train.npy` contains numpy array of dataset. Here is an example of data structure:
        ```python
        # Format: [{user_id}, {tag_id}, {label}]
        # Label: '1' means the user uses the hashtag, and '0' means otherwise.
        array([[0, 1, 1], ..., [0, 1, 0]])
        ```

    * `networks.json` contains a list of structures for each network. Here is an example of data structure:
        ```json
        [
            {
                "name": "post",
                "src_type": "tag",
                "tgt_type": "user",
                "adj": "mangnn/data/twitter/networks/post.npy",
                "agg_src": true,
                "agg_tgt": true
            }
        ]
        ```

    * `post.npy` contains a numpy array of indices, values, and size to create a sparse tensor for an adjacency matrix that represents connections in the network. Here is an example of data structure:
        ```python
        # Format: [{indices}, {values}, {size}]
        array([array([[0, 1], [0, 3], [1, 0], [1, 3]]), # indices
               array([1, 1, 1, 1]),                     # values
               array([5, 5])])                          # size
        ```
    </details>

* **PACBERT**:  
    Prepare datasets and organize them as follows: 
    ```
    data
    └─ twitter
        └─ twitter_train.json
        └─ twitter_val.json
        └─ twitter_test.json
        └─ tag.txt
    ```
    
    <details>
    <summary>Here is an example of data structure:</summary>
    
    ```json
    [
        {
            "user": 0,
            "text": "the way to get started is to quit talking and begin doing.",
            "tag": ["life", "inspire", "goal"]
        }
    ]
    ```
    </details>

-----------------------------------------------------------

## :rocket: Model Training

### :sparkles: Non-Distributed Training
You can train the model by using `run.sh`.

* **MANGNN**:
    ```bash
    mangnn/scripts/run.sh
    ```

* **PACBERT**:
    ```bash
    pacbert/scripts/run.sh
    ```

<details>
<summary>You can also parse arguments to the script:</summary>

```bash
{MODEL_NAME}/scripts/run.sh [$CONFIG]
```

where:
* `$CONFIG` - Configuration path.
</details>


### :sparkles: Distributed Training on Single Node
You can perform distributed training on a single node by using `run_single.sh`.

* **MANGNN**:
    ```bash
    mangnn/scripts/run_single.sh
    ```

* **PACBERT**:
    ```bash
    pacbert/scripts/run_single.sh
    ```

<details>
<summary>You can also parse arguments to the script:</summary>

```bash
{MODEL_NAME}/scripts/run_single.sh [$NUM_TRAINERS] [$CONFIG]
```

where:
* `$NUM_TRAINERS` - Number of GPUs/CPUs.
* `$CONFIG` - Configuration path.
</details>


### :sparkles: Distributed Training on Multiple Nodes
You can perform distributed training on multiple nodes by using `run_multi.sh`.

* **MANGNN**:
    ```bash
    mangnn/scripts/run_multi.sh
    ```

* **PACBERT**:
    ```bash
    pacbert/scripts/run_multi.sh
    ```

<details>
<summary>You can also parse arguments to the script:</summary>

```bash
{MODEL_NAME}/scripts/run_multi.sh [$NUM_NODES] [$NUM_TRAINERS] [$NODE_RANK] [$MASTER_ADDR] [$MASTER_PORT] [$CONFIG]
```

where:
* `$NUM_NODES` - Number of machines.
* `$NUM_TRAINERS` - Number of GPUs/CPUs.
* `$NODE_RANK` - Global rank.
* `$MASTER_ADDR` - Master address.
* `$MASTER_PORT` - Master port.
* `$CONFIG` - Configuration path.

For example, running on multiple GPUs across 2 nodes.

* On master node with 2 GPUs:
    
    ```bash
    pacbert/scripts/run_multi.sh 2 2 0 123.456.789 1234
    ```

* On worker node with 4 GPUs:
    
    ```bash
    pacbert/scripts/run_multi.sh 2 4 1 123.456.789 1234
    ```
</details>

-----------------------------------------------------------

## :pushpin: Checkpoint and Logging
After training, the following files are created in folder `{MODEL_NAME}/outputs/`:
* `ckpt.pt` - Model checkpoint.
* `logs/` - Tensorboard logs.
* `result.json` - Training results containing train_loss, val_loss, and metrics.

-----------------------------------------------------------

## :star: Citation
If you find our work useful for your research, please cite the following paper:

```bibtex
@ARTICLE{9984162,
  author={Padungkiatwattana, Umaporn and Maneeroj, Saranya},
  journal={IEEE Access}, 
  title={PAC-MAN: Multi-Relation Network in Social Community for Personalized Hashtag Recommendation}, 
  year={2022},
  volume={10},
  number={},
  pages={131202-131228},
  doi={10.1109/ACCESS.2022.3229082}}
```
