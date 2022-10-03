# yolov7-triton-deepstream

## Perpare weight model and run docker 
Perpare weight:
```bash 
* see README into folder weights
* or download onnx converted from link drive
```
Pepare docker: 
```bash 
* docker pull thanhlnbka/deepstream-python-app:3.0-triton 
* bash run_docker.sh 
```
##### NOTE: NEXT STEPS WORK INTO DOCKER
## Using deepstream-triton to convert engine
Install package TensorRT: 
```bash 
* apt-get install libnvinfer8 libnvinfer-plugin8 libnvparsers8 libnvonnxparsers8 libnvinfer-bin libnvinfer-dev libnvinfer-plugin-dev libnvparsers-dev libnvonnxparsers-dev libnvinfer-samples libcudnn8-dev libcudnn8 
* apt-mark hold libnvinfer* libnvparsers* libnvonnxparsers* libcudnn8* tensorrt 
```

Export Engine:
```bash 
* cd /opt/nvidia/deepstream/deepstream-6.1/sources/yolov7-triton-deepstream/weights
* /usr/src/tensorrt/bin/trtexec --onnx=yolov7-tiny.onnx --minShapes=images:1x3x640x640 --optShapes=images:8x3x640x640 --maxShapes=images:8x3x640x640 --fp16 --workspace=4096 --saveEngine=yolov7-fp16-1x8x8.engine --timingCacheFile=timing.cache 
* mv yolov7-fp16-1x8x8.engine ../triton_yolov7/yolov7/1/model.plan
```
## Custom config parse box 
Run cmd:
```bash 
* cd /opt/nvidia/deepstream/deepstream-6.1/sources/yolov7-triton-deepstream/nvdsinfer_custom_impl_Yolo

* CUDA_VER=11.7 make install
```
## Demo 
Run demo:
```bash 
* cd /opt/nvidia/deepstream/deepstream-6.1/sources/yolov7-triton-deepstream
* python3 demo.py -i rtsp://localhost:128/gst -g nvinferserver -c configs/config_infer_triton_yolov7.txt
```
![Screenshot from 2022-10-03 14-31-10](https://user-images.githubusercontent.com/56015771/193529486-2609b621-12d8-4390-8092-a42f76bd3cd5.png)


## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/NVIDIA-AI-IOT/deepstream_python_apps](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)

</details>
