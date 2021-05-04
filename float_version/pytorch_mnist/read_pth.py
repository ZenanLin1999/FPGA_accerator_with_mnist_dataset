import os
import torch
from model import Net
import matplotlib.pyplot as plt
import numpy as np

# create folder to contain the weigth & bias
path = "./weight_bias_separate"
path = path.strip()
isExists = os.path.exists(path)
if not isExists:
    os.makedirs(path)

# create model
model = Net()
# load model weights
model_weight_path = "./mnist.pth"
model.load_state_dict(torch.load(model_weight_path))
print(model)

weights_keys = model.state_dict().keys()

# [kernel_number, kernel_channel, kernel_height, kernel_width]
# conv1-w   32*1*3*3 K*CH*W*H
conv1_0_weight_t = model.state_dict()['conv1.0.weight'].numpy()
# K*CH*W*H => H*W*CH*K
conv1_0_weight_t = np.transpose(conv1_0_weight_t, (2, 3, 1, 0))
conv1_0_weight_t = np.reshape(conv1_0_weight_t, (32*1*3*3, 1))    # 转换为2维数组
np.savetxt("./weight_bias_separate/conv1_0_weight_t.dat", conv1_0_weight_t, fmt='%f', delimiter=',')

# conv1-b   32     CH
conv1_0_bias_t = model.state_dict()['conv1.0.bias'].numpy()
conv1_0_bias_t = np.reshape(conv1_0_bias_t, (32, 1))    # 转换为2维数组
np.savetxt("./weight_bias_separate/conv1_0_bias_t.dat", conv1_0_bias_t, fmt='%f', delimiter=',')

# conv2-w   64*32*3*3 CH*W*H
conv2_0_weight_t = model.state_dict()['conv2.0.weight'].numpy()
# K*CH*W*H => H*W*CH*K
conv2_0_weight_t = np.transpose(conv2_0_weight_t, (2, 3, 1, 0))
conv2_0_weight_t = np.reshape(conv2_0_weight_t, (64*32*3*3, 1))    # 转换为2维数组
np.savetxt("./weight_bias_separate/conv2_0_weight_t.dat", conv2_0_weight_t, fmt='%f', delimiter=',')

# conv2-b   64     CH
conv2_0_bias_t = model.state_dict()['conv2.0.bias'].numpy()
conv2_0_bias_t = np.reshape(conv2_0_bias_t, (64, 1))    # 转换为2维数组
np.savetxt("./weight_bias_separate/conv2_0_bias_t.dat", conv2_0_bias_t, fmt='%f', delimiter=',')

# conv3-w   64*64*3*3 CH*W*H
conv3_0_weight_t = model.state_dict()['conv3.0.weight'].numpy()
# K*CH*W*H => H*W*CH*K
conv3_0_weight_t = np.transpose(conv3_0_weight_t, (2, 3, 1, 0))
conv3_0_weight_t = np.reshape(conv3_0_weight_t, (64*64*3*3, 1))    # 转换为2维数组
np.savetxt("./weight_bias_separate/conv3_0_weight_t.dat", conv3_0_weight_t, fmt='%f', delimiter=',')

# conv3-b   64     CH
conv3_0_bias_t = model.state_dict()['conv3.0.bias'].numpy()
conv3_0_bias_t = np.reshape(conv3_0_bias_t, (64, 1))    # 转换为2维数组
np.savetxt("./weight_bias_separate/conv3_0_bias_t.dat", conv3_0_bias_t, fmt='%f', delimiter=',')

# dense1-w  128*576(64*3*3)
dense1_0_weight_t = model.state_dict()['dense.0.weight'].numpy()
dense1_0_weight_t = np.reshape(dense1_0_weight_t, (128, 64, 3, 3))    # 转换为4维数组
# K*CH*W*H => H*W*CH*K
dense1_0_weight_t = np.transpose(dense1_0_weight_t, (2, 3, 1, 0))
dense1_0_weight_t = np.reshape(dense1_0_weight_t, (128*576, 1))    # 转换为2维数组
np.savetxt("./weight_bias_separate/dense1_0_weight_t.dat", dense1_0_weight_t, fmt='%f', delimiter=',')

# dense1-b  128
dense1_0_bias_t = model.state_dict()['dense.0.bias'].numpy()
dense1_0_bias_t = np.reshape(dense1_0_bias_t, (128, 1))    # 转换为2维数组
np.savetxt("./weight_bias_separate/dense1_0_bias_t.dat", dense1_0_bias_t, fmt='%f', delimiter=',')

# dense2-w  10*128
dense2_0_weight_t = model.state_dict()['dense.2.weight'].numpy()
dense2_0_weight_t = np.reshape(dense2_0_weight_t, (10, 128, 1, 1))    # 转换为4维数组
# K*CH*W*H => H*W*CH*K
dense2_0_weight_t = np.transpose(dense2_0_weight_t, (2, 3, 1, 0))
dense2_0_weight_t = np.reshape(dense2_0_weight_t, (10*128, 1))    # 转换为2维数组
np.savetxt("./weight_bias_separate/dense2_0_weight_t.dat", dense2_0_weight_t, fmt='%f', delimiter=',')

# dense2-b  10
dense2_0_bias_t = model.state_dict()['dense.2.bias'].numpy()
dense2_0_bias_t = np.reshape(dense2_0_bias_t, (10, 1))    # 转换为2维数组
np.savetxt("./weight_bias_separate/dense2_0_bias_t.dat", dense2_0_bias_t, fmt='%f', delimiter=',')
