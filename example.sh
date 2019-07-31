# set up folder
mkdir example_result

# get data
wget -P example_result http://kaldir.vc.in.tum.de/3dsis/scannet_benchmark_example_data.zip
unzip example_result/scannet_benchmark_example_data.zip -d example_result

# get enet checkpoint
wget -P example_result http://kaldir.vc.in.tum.de/3dsis/scannet_enet_checkpoint.zip
unzip example_result/scannet_enet_checkpoint.zip -d example_result

# get 3d-sis model checkpoint
wget -P example_result http://kaldir.vc.in.tum.de/3dsis/scannet_benchmark_checkpoint.zip
unzip example_result/scannet_benchmark_checkpoint.zip -d example_result
mkdir example_result/scannet_benchmark_checkpoint/ScanNet
mkdir example_result/scannet_benchmark_checkpoint/ScanNet/example
mkdir example_result/scannet_benchmark_checkpoint/ScanNet/example/checkpoints
mv example_result/scannet_benchmark_checkpoint/step_* example_result/scannet_benchmark_checkpoint/ScanNet/example/checkpoints

# inference
python main.py --cfg ScanNet/example --step 1205541 --output_dir example_result/scannet_benchmark_checkpoint --mode benchmark --gpu 0

# from voxel(sdf) to mesh vertices
echo 'Voxel to Mesh'
python tools/scannet_benchmark/vox2mesh.py --pred_dir example_result/test --output_dir example_result/benchmark_result --scan_path example_result/scannet_benchmark_example_data/scans --frames example_result/scannet_benchmark_example_data/images_square

# generate visualization
echo 'generate visualization'
python tools/scannet_benchmark/visualize_benchmark.py --output_dir example_result/visualization --result_dir example_result/benchmark_result  --scan_path example_result/scannet_benchmark_example_data/scans

# evaluate using scannet benchmark evaluation script
echo 'evaluate mAP'
python tools/scannet_benchmark/evaluate_semantic_instance.py --pred_path example_result/benchmark_result --gt_path example_result/scannet_benchmark_example_data/gt_insts --output_file example_result/bencmark_result.txt

