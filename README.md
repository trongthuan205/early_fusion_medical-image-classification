<div align="center">
<img src="resources/mmcls-logo.png" width="600"/>
</div>

## Train Individual Deep Learning Model (Deep Doctor)
### Installation MMClassification

Below are quick steps for installation:

```shell
conda create -n open-mmlab python=3.8 pytorch=1.10 cudatoolkit=11.3 torchvision==0.11.0 -c pytorch -y
conda activate open-mmlab
pip3 install openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmclassification.git
cd mmclassification
pip3 install -e .
```

Colab tutorials are also provided:

- Learn about MMClassification **Python API**: [Preview the notebook](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_python.ipynb) or directly [run on Colab](https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_python.ipynb).
- Learn about MMClassification **CLI tools**: [Preview the notebook](https://github.com/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_tools.ipynb) or directly [run on Colab](https://colab.research.google.com/github/open-mmlab/mmclassification/blob/master/docs/en/tutorials/MMClassification_tools.ipynb).

### Training

The configurations for DenseNet-201 and T2T-ViT models, as well as the pre-trained ChestXray and Clean-CC-CCII models, are located in the "configs/" directory, "ckpt_chestxray/", and "ckpt_ccii/" directories, respectively. In order to train, it is advised that the path to the dataset and class file be adjusted in the configuration files. It is worth noting that the training was conducted on two GPUs.

```shell
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 bash ./tools/dist_train.sh ${CONFIG_FILE} 2
```

### Testing
The model weight is automatically saved according to the path specified in the "work_dir" parameter of the configuration file.

```shell
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 bash ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} 2 --metrics accuracy --metric-options topk=1
```

### Early Fusion

## Acknowledgement

MMClassification is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new classifiers.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2020mmclassification,
    title={OpenMMLab's Image Classification Toolbox and Benchmark},
    author={MMClassification Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmclassification}},
    year={2020}
}
```

