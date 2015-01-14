import cv2
import numpy
import pycuda.driver as cuda

import libcudnn_rc2 as cudnn

# For float (works correctly):
# numpy_data_type = numpy.float32
# cudnn_data_type = cudnn.cudnnDataType["CUDNN_DATA_FLOAT"]

# For double (gives some strange values)
numpy_data_type = numpy.float64
cudnn_data_type = cudnn.cudnnDataType["CUDNN_DATA_DOUBLE"]


tensor_format = cudnn.cudnnTensorFormat["CUDNN_TENSOR_NCHW"]
conv_mode = cudnn.cudnnConvolutionMode["CUDNN_CROSS_CORRELATION"]
conv_algo = cudnn.cudnnConvolutionFwdAlgo["CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM"]


# CUDA init
cuda.init()
cur_device = cuda.Device(0)
cur_context = cur_device.make_context()


handle = cudnn.cudnnCreate()

src_desc = cudnn.cudnnCreateTensorDescriptor()

# Src pic
n_pics = 1
n_chans = 3
pic_h = 500
pic_w = 400

src_pic = cv2.split(cv2.imread("FM.jpg"))
src_pic = numpy.array(src_pic, dtype=numpy_data_type).reshape((1, 3, 500, 400))

src_dev = cuda.to_device(src_pic)

# Filters
filter_pic = numpy.array([[[[1, 1, 1], [1, 1, 1], [1, 1, 1]]]], dtype=numpy_data_type)
filter_dev = cuda.to_device(filter_pic)

cudnn.cudnnSetTensor4dDescriptor(
    src_desc, tensor_format, cudnn_data_type, n_pics, n_chans, pic_h, pic_w)

filter_h = 3
filter_w = 3

filter_desc = cudnn.cudnnCreateFilterDescriptor()
cudnn.cudnnSetFilter4dDescriptor(filter_desc, cudnn_data_type, 1, 3, filter_h, filter_w)

# Conv
conv_desc = cudnn.cudnnCreateConvolutionDescriptor()
cudnn.cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, conv_mode)

n_out_pics, n_out_chans, out_h, out_w = cudnn.cudnnGetConvolution2dForwardOutputDim(
    conv_desc, src_desc, filter_desc)

dest_pic = numpy.zeros(
    shape=(n_out_pics, n_out_chans, out_h, out_w), dtype=numpy_data_type)
dest_dev = cuda.mem_alloc_like(dest_pic)

dest_desc = cudnn.cudnnCreateTensorDescriptor()
cudnn.cudnnSetTensor4dDescriptor(dest_desc, tensor_format, cudnn_data_type,
                                 n_out_pics, n_out_chans, out_h, out_w)

alpha = 1.
beta = 0.
cudnn.cudnnConvolutionForward(
    handle, alpha, src_desc, int(src_dev), filter_desc, int(filter_dev),
    conv_desc, conv_algo, 0, 0, beta, dest_desc, int(dest_dev))

# Reading output pic:
dest_pic = cuda.from_device_like(dest_dev, dest_pic)
pic_to_show = numpy.array(dest_pic[0][0]) / 255. / 9

print(pic_to_show * 255) # Should be from 0 to 255
cv2.imshow("FMD", pic_to_show) # Here should be a slightly blurred pic
cv2.waitKey(0)

cur_context.pop()