#!/usr/bin/env bash

export PATH=$PATH:/home/user/.local/bin

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export nnUNet_raw=resources/input
export nnUNet_results=resources/second_nnunet_result
export nnUNet_preprocessed=resources/nnunet_preprocessed

nnUNetv2_install_pretrained_model_from_zip resources/nnUNetTrainer__nnUNetPlans__3d_fullres_all_woblmask_clean.zip

## for more accurate inference, use:
#nnUNetv2_predict -i test/input/nnunet_raw/Dataset001_HS/ImagesTs -o resources -d '001' -c 3d_fullres

## for faster, less accurate inference use:
nnUNetv2_predict -step_size 0.9 --disable_tta -npp 5 -device 'cuda' -i resources/input -o resources/second_nnunet_result -d '001' -c 3d_fullres