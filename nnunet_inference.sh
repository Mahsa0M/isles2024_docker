#!/usr/bin/env bash

export PATH=$PATH:/home/user/.local/bin

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# setting the paths necessary for nnunet
export nnUNet_raw=resources/input
export nnUNet_results=resources/nnunet_result
export nnUNet_preprocessed=resources/nnunet_preprocessed

# installing the trained nnuent model
nnUNetv2_install_pretrained_model_from_zip resources/nnUNetTrainer__nnUNetPlans__3d_fullres_all_woblmask_clean.zip

## for more accurate inference, use:
#nnUNetv2_predict -i resources/input -o resources/nnunet_result -d '001' -c 3d_fullres

## for faster, less accurate inference use:
nnUNetv2_predict -step_size 0.9 --disable_tta -npp 5 -device 'cuda' -i resources/input -o resources/nnunet_result -d '001' -c 3d_fullres