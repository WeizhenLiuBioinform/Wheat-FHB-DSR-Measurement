## Training

python train.py --workers 8 --device 0 --batch 8 --data /data/Wheat_FHB_DSR_Measurement_dataset.yaml --img 640 --cfg /models/detect/fhbdsr-net.yaml --weights '/gelan-c.pt' --hyp hyp.scratch-high.yaml --epochs 150

## Validation

python val.py --data /data/Wheat_FHB_DSR_Measurement_dataset.yaml --weights './trained_fhbdsr_net.pt' --task val --device 0 


## Inference

python detect.py --source './data/images/Data_sample_1.jpg' --img 640 --device 0 --weights './trained_fhbdsr_net.pt' --name spikelet_detect