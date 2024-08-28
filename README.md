The algorithm was submitted to ISLES2024 challenge. 
The branch one_unet contains the code for an nnU-Net trained with the NCCT, CTA and CTP maps.
For preprocessing, we first remove the background and then clip the values of all the scans.
The trained nnU-Net weights can be found in the drive: https://drive.google.com/file/d/1kAFSgQdMpJ4HQQ9wZWiWGXgr7_rUyP_K/view?usp=drive_link
In order to pass the time requirement of the challegne, the accuracy of the inference had to be lowered. In case you want to use the higher quality inference, uncomment the line in nnunet_inference.sh that refers to the more accurate inference.