The algorithm that was submitted to ISLES2024 challenge.

## Method Description
Our final infarct segmentation model is an [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). We used the 3D full-resolution setting of nnU-Net version 2. Our submitted model is trained on the first two batches of the [ISLES24 dataset](https://isles-24.grand-challenge.org/).

We included the following inputs to our model:
1. non-contrast CT (NCCT)
2. CT Angiography (CTA)
3. relative cerebral blood flow (rCBF)
4. relative cerebral blood volume (rCBV)
5. mean transit time (MTT)
6. time to maximum (Tmax). 

### Preprocessing
The scans were pre-processed by registering to NCCT (provided by ISLES), removing the background, and clipping the values as follows: NCCT: [0, 100] HU, CTA: [0, 200] HU, rCBF: [0, 400] %, rCBV: [0, 400] %, MTT: [0, 20] s, and Tmax: [0, 20] s. During nnU-Net preprocessing, we apply global normalization for NCCT and CTA, and per-channel z-scoring for the CTP maps.

**NOTE:** In order to pass the time requirement of the challegne, we reduced inference time by disabling nnU-Net’s test-time data augmentation and increasing the patch inference step size to 0.9. However, according to the nnU-Net documentation, this can reduce the accuracy of inference. In case you want to use the default nnU-Net inference, uncomment the line in `nnunet_inference.sh` that refers to the more accurate inference.

## How To Use
If you are willing to upload your data to the grand-challenge website, you can use [our algorithm](https://grand-challenge.org/algorithms/one_unet_all_inputs/) there without any need for installation. Otherwise, if you want to install the docker on your own computer using this repository, you can take the following steps.

**Note:** The docker file is designed for Linux environment. In case you are working on a Windows system, you need to install Windows subsystem for Linux (WSL) first. A guide can be found [here](https://learn.microsoft.com/en-us/windows/wsl/install). Also, you need to install Docker desktop ([guide](https://docs.docker.com/desktop/install/windows-install/). On Linux, you need to install the Docker engine for Linux (guide for Ubuntu [here](https://docs.docker.com/engine/install/ubuntu/). 

1. Install this repository.
2. Download the trained nnU-Net weights from [here](https://drive.google.com/file/d/1kAFSgQdMpJ4HQQ9wZWiWGXgr7_rUyP_K/view?usp=drive_link).
3. Place the zipped model in the `resources` folder.
4. Place the co-registered scans in the `test/input/images` folder in their corresponding folders. There should be only one image per folder (this Docker is designed to handle one patient at a time). The images should be of `.mha` format.
5. Go to the folder location containing this repository in terminal (or WSL terminal if you are using Windows).
6. Run `sh test_run.Unix.sh`.
7. If no error appears, the output should be placed in `test/output` folder.
