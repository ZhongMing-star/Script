cp -r include/* /usr/local/include/
cp lib/libonnxruntime*.so* /usr/local/lib/
sudo ldconfig
# 查看库是否存在
ldconfig -p | grep onnxruntime
# 查看头文件
ls /usr/local/include | grep onnxruntime


echo 'export CMAKE_PREFIX_PATH=/usr/local/lib/cmake:$CMAKE_PREFIX_PATH' >> ~/.bashrc
source ~/.bashrc
