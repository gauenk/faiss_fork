#!/bin/bash
make -C build -j swigfaiss
cd /home/gauenk/Documents/faiss/build/faiss/python
python ./setup.py install --user
cd /home/gauenk/.local/lib/python3.8/site-packages
unzip -o faiss-1.7.1-py3.8.egg
cd /home/gauenk/Documents/faiss/
