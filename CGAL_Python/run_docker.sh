docker run \
  -dit \
  --gpus all \
  --name="build_py_package" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd):/home/lz/data \
  -e DISPLAY=:0 \
  -e QT_X11_NO_MITSHM=1 \
  fxiangucsd/sapien-build-env \
  bash

docker exec -it build_py_package bash
