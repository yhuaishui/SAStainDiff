# SAStainDiff
[BSPC] The official code of ["SAStainDiff: Self-supervised Stain Normalization by Stain Augmentation using Denoising Diffusion Probabilistic Models"](https://github.com/yhuaishui/SAStainDiff/).

In this paper, we propose a new self-supervised diffusion probabilistic modeling approach for stain normalization with a rescheduled sampling strategy.
<p align="center">
<img src=assets/fig1.png />
</p>

## Models Trained by Us
Models (normalized to Aperio/Hamamatsu domain) trained by us can be downloaded from Hugging Face: [pretrained model](https://huggingface.co/yhuaishui/SAStainDiff/tree/main).

## Dataset
- The MITOS-ATYPIA’14 Challenge dataset can be downloaded from here: [MITOS-ATYPIA’14](https://mitos-atypia-14.grand-challenge.org/).
- The subset of Camelyon16 can be downloaded from here: [Camelyon16](https://pan.baidu.com/s/1_k7l3wL0vrP26Yc6kkcWEQ#list/path=%2F) Extraction code：wrfi; More details on this dataset can be found in [StainNet](https://github.com/khtao/StainNet).
- The MICCAI’15 GlaS dataset can be downloaded from here: [GlaS](https://academictorrents.com/details/208814dd113c2b0a242e74e832ccac28fcff74e5).

## Stain Augmentation
The Stain Database can be downloaded from Hugging Face: [Stain Database](https://huggingface.co/yhuaishui/SAStainDiff/resolve/main/stain_database.pickle?download=true).

To extend the stain database with new histopathology patches, run:
```database
python stain_augmentation/extend_kd_tree_offline.py -i <input pickle file, database to extend> -o <output pickle file> -d <data path, patches to extend database>
```
In particular, we performed stain augmentation referring to [Data_Driven_Color_Augmentation](https://github.com/ilmaro8/Data_Driven_Color_Augmentation).

## Usage

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), and you can refer to its settings for running.

This repository is implemented and tested on Python 3.8 and PyTorch 1.13.1.
To install requirements:

```setup
pip install -r requirements.txt
```

To train the model, modify the settings in the train.sh, and run:

```train
bash train.sh
```
- data_dir <Input_data_path>
- stain_database_path <Input_database_path>
- checkpoint_path <Output_checkpoint_path>
- resume_checkpoint <checkpoint_model_path_for_continue_training>

To test the model, modify the settings in the test.sh, and run:

```test
bash test.sh
```
- input_path <Input_data_path>
- save_path <Output_data_path>
- model_path your <checkpoint_model_path>
- timestep_step <the_initial_sampling_point>
- timestep_respacing <the_sampling_step_size>
- use_anysize <Restore_Any-size_Image_sampling_strategy>
- use_ddim <DDIM_sampling_strategy>
  
## Citation

## Feedback

For any feedback or inquiries, please contact yanghuaishui@foxmail.com
