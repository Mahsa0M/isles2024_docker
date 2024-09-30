"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To export the container and prep it for upload to Grand-Challenge.org you can call:

  docker save example-algorithm-preliminary-docker-evaluation | gzip -c > example-algorithm-preliminary-docker-evaluation.tar.gz

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
import shutil
from pathlib import Path
from glob import glob
import SimpleITK as sitk
import numpy as np
import os
from scipy.ndimage.morphology import binary_fill_holes


def run():
    INPUT_PATH = Path("/input")
    OUTPUT_PATH = Path("/output")
    RESOURCE_PATH = Path("resources")

    # Read input data.
    ncct = load_image_file_as_array(
        location=INPUT_PATH / "images/non-contrast-ct",
    )
    registered_cta = load_image_file_as_array(
        location=INPUT_PATH / "images/preprocessed-CT-angiography",
    )
    registered_tmax = load_image_file_as_array(
        location=INPUT_PATH / "images/preprocessed-tmax-map",
    )
    registered_cbf = load_image_file_as_array(
        location=INPUT_PATH / "images/preprocessed-cbf-map",
    )
    registered_cbv = load_image_file_as_array(
        location=INPUT_PATH / "images/preprocessed-cbv-map",
    )
    registered_mtt = load_image_file_as_array(
        location=INPUT_PATH / "images/preprocessed-mtt-map",
    )

    # Preprocess the input data
    preprocessed_ncct = preprocess_scan(ncct, clip_min=0, clip_max=100)
    preprocessed_cta = preprocess_scan(registered_cta, clip_min=0, clip_max=200)
    preprocessed_tmax = preprocess_scan(registered_tmax, clip_min=0, clip_max=20)
    preprocessed_cbf = preprocess_scan(registered_cbf, clip_min=0, clip_max=400)
    preprocessed_cbv = preprocess_scan(registered_cbv, clip_min=0, clip_max=400)
    preprocessed_mtt = preprocess_scan(registered_mtt, clip_min=0, clip_max=20)

    # create the raw data and result folders for nnunet inference
    nnunet_raw_data_path = RESOURCE_PATH / 'input'
    if not os.path.exists(nnunet_raw_data_path):
        os.mkdir(nnunet_raw_data_path)
        os.mkdir(RESOURCE_PATH / 'nnunet_preprocessed')

    # move the preprocessed images to the raw data folder for nnunet inference
    sitk.WriteImage(preprocessed_ncct, nnunet_raw_data_path / 'isles_0001_0000.nii.gz')
    sitk.WriteImage(preprocessed_cta, nnunet_raw_data_path / 'isles_0001_0001.nii.gz')
    sitk.WriteImage(preprocessed_cbf, nnunet_raw_data_path / 'isles_0001_0002.nii.gz')
    sitk.WriteImage(preprocessed_cbv, nnunet_raw_data_path / 'isles_0001_0003.nii.gz')
    sitk.WriteImage(preprocessed_tmax, nnunet_raw_data_path / 'isles_0001_0004.nii.gz')
    sitk.WriteImage(preprocessed_mtt, nnunet_raw_data_path / 'isles_0001_0005.nii.gz')


    # prediction
    stroke_lesion_segmentation = predict_infarct(RESOURCE_PATH)

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/stroke-lesion-segmentation",
        array=stroke_lesion_segmentation.astype(np.int32),
    )

    return 0


def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    # print(glob(str(location / '*')))
    input_files = glob(str(location / "*.mha"))
    result = sitk.ReadImage(input_files[0])
    return result
    # Convert it to a Numpy array
    # return sitk.GetArrayFromImage(result)


def write_array_as_image_file(*, location, array):
    location.mkdir(parents=True, exist_ok=True)

    suffix = ".mha"
    # print(str(location / f"output{suffix}"))
    image = sitk.GetImageFromArray(array)
    # print(sum(image))
    sitk.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )


def predict_infarct(RESOURCE_PATH):
    """
    Runs a file with the necessary commands to perform the inference using nnunet
    RESOURCE_PATH: where nnunet results will be saved.
    """
    # inference
    os.popen('sh nnunet_inference.sh').read()

    # read the prediction
    pred = sitk.ReadImage(RESOURCE_PATH / 'nnunet_result' / 'isles_0001.nii.gz')
    pred_np = sitk.GetArrayFromImage(pred)

    return pred_np


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


##################################### functions added by me #####################################
def preprocess_scan(scan, clip_min, clip_max):
    """
    clip range for ncct: [0, 100]
    clip range for cta: [0, 200]
    """
    scan = remove_background(scan)  # remove background
    scan = Clipper(scan, clip_min, clip_max)  # clip values
    return scan

### supporting functions
def remove_background(sitk_image):
    image_no_background_mask = LargestConnectedComponent3D(sitk_image)
    image_no_background_mask = np_slicewise(image_no_background_mask, [binary_fill_holes])
    out = ApplyMask(image_no_background_mask, sitk_image)
    return out

def Clipper(scan, minimum=-1024, maximum=1900):
    """
    Clip values of input with numpy.clip, with minimum and maximum as min/max.
    Returns clipped array.

    """
    scan_array = sitk.GetArrayFromImage(scan) # T x D x H x W
    scan_array = np.clip(scan_array, minimum, maximum)
    scan_sitk = np2itk(scan_array, scan)
    return scan_sitk

def LargestConnectedComponent3D(img,min_threshold=0, background=0):
    """
    Retrieves largest connected component mask for 3D sitk
    """
    # compute connected components (in 3D)
    cc = sitk.ConnectedComponent(img>min_threshold)
    stats = sitk.LabelIntensityStatisticsImageFilter()
    stats.Execute(cc,img)
    max_size = 0
    # get largest connected component
    for l in stats.GetLabels():
        if stats.GetPhysicalSize(l)>max_size:
            max_label = l
            max_size = stats.GetPhysicalSize(l)
    # return mask
    return sitk.BinaryThreshold(cc, max_label, max_label+1e-2)

def np_slicewise(mask, funcs, repeats=1):
    """
    Fills holes slice by slice of an 3D np volume
    """
    original = mask
    if isinstance(mask,sitk.SimpleITK.Image):
        mask = sitk.GetArrayFromImage(mask)
    out = np.zeros_like(mask)
    for sliceno in range(mask.shape[0]):
        m = mask[sliceno,:,:]
        for r in range(repeats):
            for func in funcs:
                m = func(m)
        out[sliceno,:,:] = m
    out = np2itk(out, original)
    return out

def np2itk(arr, original_img):
    if len(arr.shape) == 4:
        t_dim = arr.shape[0]
        frames = []
        for t in range(t_dim):
            frames.append(sitk.GetImageFromArray(arr[t], False))
        img = sitk.JoinSeries(frames)
    elif len(arr.shape) == 3:
        img = sitk.GetImageFromArray(arr, False)

    img.SetSpacing(original_img.GetSpacing())
    img.SetOrigin(original_img.GetOrigin())
    img.SetDirection(original_img.GetDirection())
    # this does not allow cropping (such as removing thorax, neck)
    img.CopyInformation(original_img)
    return img

def ApplyMask(mask, scan, foreground_m=1, background=-1024, sitk_type=sitk.sitkUInt8):
    """
    Applies mask (m) to 3D volume, sets background of image
    returns and 4D volume with only mask foreground.
    """
    if foreground_m == 0:
        mf = sitk.MaskNegatedImageFilter()
    elif foreground_m == 1:
        mf = sitk.MaskImageFilter()
    if background != None:
        mf.SetOutsideValue(background)
    scan = sitk.Cast(scan, sitk_type)
    mask = sitk.Cast(mask, sitk_type)

    assert np.allclose(scan.GetOrigin(), mask.GetOrigin(), atol=0.01)
    assert np.allclose(scan.GetSpacing(), mask.GetSpacing(), atol=0.01)

    mask.SetOrigin(scan.GetOrigin())
    mask.SetSpacing(scan.GetSpacing())

    result = mf.Execute(scan, mask)

    return result

if __name__ == "__main__":
    raise SystemExit(run())
