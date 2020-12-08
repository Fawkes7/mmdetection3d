#!/usr/bin/env bash
function build_manylinux14_wheel() {
  PY_VERSION=$1
  echo "Start deal with python verion $PY_VERSION now"
  if [ "$PY_VERSION" -eq 36 ]; then
      PY_DOT=3.6
      EXT="m"
  elif [ "$PY_VERSION" -eq 37 ]; then
      PY_DOT=3.7
      EXT="m"
  elif [ "$PY_VERSION" -eq 38 ]; then
      PY_DOT=3.8
      EXT=""
  elif [ "$PY_VERSION" -eq 39 ]; then
      PY_DOT=3.9
      EXT=""
  else
    echo "Error, python version not found!"
  fi

  INCLUDE_PATH=/usr/include/python${PY_DOT}
  BIN=/usr/bin/python${PY_DOT}
  echo "Using bin path ${BIN}"
  echo "Using include path ${INCLUDE_PATH}"
  export CPLUS_INCLUDE_PATH=$INCLUDE_PATH

  COMMAND="${BIN} setup.py bdist_wheel"


  echo "Running command ${COMMAND}"
  eval "$COMMAND"

  WHEEL_NAME="./dist/pycgal-1.0-cp${PY_VERSION}-cp${PY_VERSION}${EXT}-linux_x86_64.whl"
  if test -f "$WHEEL_NAME"; then
    echo "$FILE exist, begin audit and repair"
  fi
  WHEEL_COMMAND="auditwheel repair ${WHEEL_NAME}"
  eval "$WHEEL_COMMAND"
}

rm -rf build
mkdir build
build_manylinux14_wheel 36
build_manylinux14_wheel 37
build_manylinux14_wheel 38
build_manylinux14_wheel 39
rm -rf build

