./bootstrap.sh --with-python=/usr/bin/python3.6 --with-python-version=3.6 --with-python-root=/usr/lib/ --prefix=/usr
sudo update-alternatives --install /usr/bin/python python3.6 /usr/bin/python3.6 36
./b2 install --prefix=/usr --python-buildid=36  --with-python
./b2 install --prefix=/usr --python-buildid=py36 --with-python


./bootstrap.sh --with-python=/usr/bin/python3.7 --with-python-version=3.7 --with-python-root=/usr/lib/python3.7 --prefix=/usr
sudo update-alternatives --install /usr/bin/python python3.7 /usr/bin/python3.7 37
./b2 install --prefix=/usr --python-buildid=37  --with-python
./b2 install --prefix=/usr --python-buildid=py37 --with-python

./bootstrap.sh --with-python=/usr/bin/python3.8 --with-python-version=3.8 --with-python-root=/usr/lib/ --prefix=/usr
sudo update-alternatives --install /usr/bin/python python3.8 /usr/bin/python3.8 38
./b2 install --prefix=/usr --python-buildid=38  --with-python
./b2 install --prefix=/usr --python-buildid=py38 --with-python

./bootstrap.sh --with-python=/usr/bin/python3.9 --with-python-version=3.9 --with-python-root=/usr/lib/ --prefix=/usr
sudo update-alternatives --install /usr/bin/python python3.9 /usr/bin/python3.9 39
./b2 install --prefix=/usr --python-buildid=39  --with-python
./b2 install --prefix=/usr --python-buildid=py39 --with-python


# Install python

#./configure --enable-optimizations --enable-shared --prefix=/usr/ LDFLAGS="-Wl,-rpath /usr/lib"

sudo update-alternatives --install /usr/bin/python python2.7 /usr/bin/python2.7 27
sudo update-alternatives --install /usr/bin/python python3.6 /usr/bin/python3.6 36
sudo update-alternatives --install /usr/bin/python python3.7 /usr/bin/python3.7 37
sudo update-alternatives --install /usr/bin/python python3.8 /usr/bin/python3.8 38
sudo update-alternatives --install /usr/bin/python python3.9 /usr/bin/python3.9 39

#Install Cmake
./bootstrap --prefix=/usr && make && sudo make install
