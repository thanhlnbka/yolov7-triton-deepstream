docker run -it --rm --gpus all --network host \
-v $PWD:/opt/nvidia/deepstream/deepstream-6.1/sources/yolov7-triton-deepstream \
-e DISPLAY=$DISPLAY -e XAUTHORITY=$XAUTHORITY -v $XAUTHORITY:$XAUTHORITY -v /tmp/.X11-unix:/tmp/.X11-unix thanhlnbka/deepstream-python-app:3.0-triton