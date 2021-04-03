# Pytorch modules
import torch
from torch import nn
import torch.nn.functional as F

# Other modules
import matplotlib.pyplot as plt



class PadWithin(nn.Module):
    """
    Zero Insertion Module
    Ref: https://github.com/pytorch/pytorch/issues/7911
    """
    def __init__(self, stride):
        super(PadWithin, self).__init__()
        self.stride = stride
        w = torch.zeros(self.stride, self.stride)
        w[0,0] = 1
        self.register_buffer('w', w) # Register a buffer so that tensor device will not be an issue

    def forward(self, feats):
        return F.conv_transpose2d(feats, self.w.expand(feats.size(1), 1, self.stride, self.stride), stride=self.stride, 
                groups=feats.size(1))


def test_pad_within(size=8):
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    pad_within = PadWithin(2).to(device)
    test_tensor = torch.arange(start=0, end=10.0, step=1/(size*size)).view((1, 10, size, size)).to(device)
    padded_tensor = pad_within(test_tensor)
    print( "test_tensor shape = {}, padded_tensor_shape = {}".format(test_tensor.shape, padded_tensor.shape) )

    _, axs = plt.subplots(1, 2)
    axs[0].imshow(test_tensor[0, 9, :, :].cpu().numpy())
    axs[1].imshow(padded_tensor[0, 9, :, :].cpu().numpy())
    plt.show()


if __name__ == "__main__":
    test_pad_within()