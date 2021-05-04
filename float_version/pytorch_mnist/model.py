import torch

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(       #input:28*28*1
            torch.nn.Conv2d(1, 32, 3, 1, 1),    #in_channels,out_channels,kernel_size,stride,padding   out_width = (input_width - kernel_size + 2*padding) / stride + 1
            torch.nn.ReLU(),                    #28*28*32
            torch.nn.MaxPool2d(2))              #14*14*32
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),   #in_channels,out_channels,kernel_size,stride,padding
            torch.nn.ReLU(),                    #14*14*64
            torch.nn.MaxPool2d(2)               #7*7*64
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 3, 1, 1),   #in_channels,out_channels,kernel_size,stride,padding
            torch.nn.ReLU(),                    #7*7*64
            torch.nn.MaxPool2d(2)               #3*3*64
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 3 * 3, 128),   #in_features,out_features,bias
            torch.nn.ReLU(),                    #128
            torch.nn.Linear(128, 10)            #10
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        res = conv3_out.view(conv3_out.size(0), -1)
        out = self.dense(res)
        return out

