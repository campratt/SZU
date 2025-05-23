# SZU

Train a machine learning model to extract the thermal Sunyaev-Zel'dovich (SZ) effect with supervised learning.

There are three parts to this pipeline:
- Data generation
- Model training
- Model evaluation

## Data generation instructions
### Obtain necessary data
- Download Planck frequency data: https://portal.nersc.gov/project/cmb/planck2020/
- Retrieve synthetic SZ data
  ```
  python3 get_Han21.py --output_dir ./Han21_output --ID 00000
  ```
### Inject/superimpose the SZ signal
- Use the results from Han et al. (2021) saved in "sz_dir" and inject them into the Planck frequency data in "freq_dir"
- Smooths the Compton-y maps with a Gaussian kernel of FWHM = 10 arcminutes, which are used as labeled data in supervised learning.
- Also smooths the SZ signals injected into each frequency map using the given resolution of each instrument.
```
python3 inject_sz_signal.py --freq_dir ./freq_dir --sz_dir ./sz_dir --ID 00000 --output_dir ./fullsky_data --cores 8
```
### Generate cutouts for training/testing machine learning model
- Creates square cutouts for the input frequency data and labeled SZ data.
- Cutouts are 512x512 with 2 arcminute pixel resolution.
- Centers of cutouts are defined by the coordinate list.
- Decompress the mask file in "utils/mask.fits.gz". The fits file masks known galaxy clusters.
- The following code can also be used to create an evaluation dataset using a custom coordinate list. An example is provided in "utils/coordinates.txt".
```
python3 generate_train_test_samples.py --data_dir ./fullsky_data --ID 00000 --output_dir ./train_test_data --coords_path ./utils/coordinates.txt --mask_path ./utils/mask.fits --cores 8
```

## Model training instructions
- The "data_dir" argument is a directory that should contain "x/", "y/", and "mask/" subdirectories. 
```
python3 main_train_szu.py --fn_config config_example.json
```

## Model evaluation instructions
- y_data_dir can be set to None when performing inference on the real sky, or set to the path for labeled simulated data
- Pretrained weights for the small model are provided in "utils/small_y0=60_weights.pth"

```
python3 main_eval_szu.py --x_data_dir ./x_eval --y_data_dir ./y_eval --model NestedUNet3D --model_size small --model_dir ./trained_models --IDs 0 1 --output_dir ./model_outputs




