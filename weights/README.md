# Export model pt to onnx with yolov7 
See https://github.com/WongKinYiu/yolov7#export for more info. <br/>

* python export.py --weights ./yolov7.pt --grid --end2end --dynamic-batch --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 <br/>


download yolov7-tiny converted onnx: https://drive.google.com/file/d/1GZbGFUk8BtEO-ZyWL2aoa3g4YS40dKiF/view?usp=sharing <br/>