## Data Generation
Data generation code is developed under VS2013, and we use [mLib](https://github.com/niessner/mLib) and [mLibExternal](http://kaldir.vc.in.tum.de/mLib/mLibExternal.zip) to generate our data, for setting up libraries, please checkout respective websites. Datasets used in this project are from [ScanNet](http://www.scan-net.org/) and [SUNCG](https://github.com/shurans/SUNCGtoolbox).

### ScanNet
* Code for ScanNet data generation is located in `ScanReal`
* Setup the paths in `ScanReal/zParameters.txt`
* Generate intermediate data for ScanNet from `ScanReal`
* Setup the paths in `SceneSampler/zParameters.txt`
* Comment out the [line](https://github.com/Sekunde/3D-SIS/blob/9e0ec054376e8baa2c3a193a9495d0f2c88c54f0/datagen/SceneSampler/GlobalAppState.h#L9) for ScanNet data.
* Generate final data including chunks used in training and scenes used in testing from `SceneSampler`
* Generate 2D images used for projection layer: enter folder `ScanReal/python` and run
```
 python prepare_2d_data.py --scannet_path /ScanNet/public/v1/scans --output_path ./scannetv2_images --label_map_file /ScanNet/public/v2/tasks/scannetv2-labels.combined.tsv
 ```

### SUCNG
* Code for ScanNet data generation is located in `SUNCGScan`
* Setup the paths in `SUNCGScan/zParameters.txt`
* Generate intermediate data and 2D images for SUNCGScan from `SUNCGScan`
* Setup the paths in `SceneSampler/zParameters.txt`
* Uncomment the [line](https://github.com/Sekunde/3D-SIS/blob/9e0ec054376e8baa2c3a193a9495d0f2c88c54f0/datagen/SceneSampler/GlobalAppState.h#L9) for SUNCG data.
* Generate final data including chunks used in training and scenes used in testing from `datagen/SceneSampler`

### Viusalizations
* You can visualize intermediate data in C++ code by setting `s_bDebugOut = true` in `zParameters.txt` file.
* You can also generate visualizations of the final data (chunks or scenes) in voxels by running
`python tools/visualization.py --path /final_data/ --mode data`. Example visualization:
<img src="http://i68.tinypic.com/2vuabyc.png" alt="3dsis" width="400"><img src="http://i66.tinypic.com/34rahzn.png" alt="3dsis" width="350">
