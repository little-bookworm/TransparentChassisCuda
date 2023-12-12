rm -r build
rm -r modules_export
mkdir build
cd build

cmake .. -DCMAKE_INSTALL_PREFIX=../modules_export
make -j6
make install

cd ..
rm -r build