#docker run -v `pwd`:/workspace/pycgal -it --rm lingzhan/python-build-env bash -c "cd /workspace/pycgal && sudo update-alternatives --install /usr/bin/python python3.6 /usr/bin/python3.6 36 && python setup.py build" #bash build_python_wheel.sh"

docker run -v `pwd`:/workspace/pycgal -it --rm lingzhan/python-build-env bash -c "cd /workspace/pycgal && bash build_python_wheel.sh"
