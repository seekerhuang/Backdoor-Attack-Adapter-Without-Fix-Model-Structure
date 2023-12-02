import torch
import torch.nn as nn
import torch.nn.functional as F

def fuse_conv_bn(conv_layer: nn.Conv2d, bn_layer: nn.BatchNorm2d) -> nn.Conv2d:
    """Fuse a convolutional layer and a batch normalization layer into a single convolutional layer."""

    # Extract weights and bias from the convolutional layer
    conv_weight = conv_layer.weight
    bias = 0

    # Extract parameters from the batch normalization layer
    gamma = bn_layer.weight
    beta = bn_layer.bias
    mean = bn_layer.running_mean
    var = bn_layer.running_var
    eps = bn_layer.eps

    # Calculate standard deviation from variance and epsilon
    std = (var + eps).sqrt()

    # Calculate the weights and bias for the fused convolutional layer
    fused_conv_weight = (gamma / std).reshape(-1, 1, 1, 1) * conv_weight
    fused_conv_bias = gamma * (bias - mean) / std + beta

    # Create a dictionary to store the parameters of the fused convolutional layer
    param_dict = {'weight': fused_conv_weight, 'bias': fused_conv_bias}

    # Create the fused convolutional layer
    fused_conv = nn.Conv2d(
        conv_layer.in_channels, conv_layer.out_channels, conv_layer.kernel_size,
        stride=conv_layer.stride, padding=conv_layer.padding, dilation=conv_layer.dilation, groups=conv_layer.groups,
    )

    # Set the parameters of the fused convolutional layer
    for name, param in fused_conv.named_parameters():
        assert param_dict.get(name) is not None
        param.data = param_dict[name]

    return fused_conv


def fuse_multi_conv_add(conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
    """Fuse the outputs of multiple convolutional branches by addition into a single convolutional layer."""

    # First, verify the consistency of the configurations in each branch
    assert conv1.in_channels == conv2.in_channels
    assert conv1.out_channels == conv2.out_channels
    assert conv1.kernel_size == conv2.kernel_size
    assert conv1.stride == conv2.stride
    assert conv1.padding == conv2.padding
    assert conv1.dilation == conv2.dilation
    assert conv1.groups == conv2.groups

    # Calculate the parameter values of the fused convolutional layer and store them in a dictionary
    fused_conv_weight = conv1.weight + conv2.weight
    fused_conv_bias = conv1.bias + conv2.bias
    param_dict = {'weight': fused_conv_weight, 'bias': fused_conv_bias}

    # Set the parameter values to the fused convolutional layer
    fused_conv = nn.Conv2d(
        conv1.in_channels, conv1.out_channels, conv1.kernel_size,
        stride=conv1.stride, padding=conv1.padding, dilation=conv1.dilation, groups=conv1.groups
    )
    for name, param in fused_conv.named_parameters():
        assert param_dict.get(name) is not None
        param.data = param_dict[name]

    return fused_conv



def fuse_sequential_conv(conv_1x1: nn.Conv2d, conv_kxk: nn.Conv2d) -> nn.Conv2d:
    """Fuse a 1x1 convolutional layer followed by a KxK convolutional layer into a single convolutional layer."""

    # Transpose the first two dimensions of the 1x1 convolutional layer
    # Note that it does not need a bias term, as the biases of the original convolutional layers will be calculated later
    transposed_conv_1x1 = nn.Conv2d(
        conv_1x1.out_channels, conv_1x1.in_channels, 1,
        stride=conv_1x1.stride, padding=conv_1x1.padding, dilation=conv_1x1.dilation, groups=conv_1x1.groups,
        bias=False
    )
    # Set the kernel of the transposed convolutional layer
    transposed_conv_1x1.weight.data = conv_1x1.weight.data.clone().transpose(0, 1)

    # Use the kernel of the KxK convolutional layer as the feature map
    kernel_kxk = conv_kxk.weight.data
    # Convolve the kernel of the KxK convolutional layer with the transposed 1x1 convolutional layer
    # The output of the convolution is the weight of the fused convolutional layer
    fused_kernel = transposed_conv_1x1(kernel_kxk)

    # Create the fused convolutional layer
    fused_conv = nn.Conv2d(
        fused_kernel.size(1), fused_kernel.size(0), fused_kernel.shape[2:],
        stride=conv_kxk.stride, padding=conv_kxk.padding, dilation=conv_kxk.dilation, groups=conv_kxk.groups
    )
    # Set the weight of the fused convolutional layer
    fused_conv.weight.data = fused_kernel
    # Calculate the bias of the fused convolutional layer based on the biases of the original convolutional layers
    fused_conv.bias.data = (conv_kxk.weight.data * conv_1x1.bias.data.reshape(1, -1, 1, 1)).sum((1, 2, 3)) \
        + conv_kxk.bias.data

    return fused_conv



def fuse_multi_linear_add(linear_1: nn.Linear, linear_2: nn.Linear) -> nn.Linear:
    """Fuse two parallel linear layers into a single linear layer by adding their outputs."""

    # Verify the consistency of the configurations of the two linear layers
    assert linear_1.in_features == linear_2.in_features
    assert linear_1.out_features == linear_2.out_features

    # Calculate the parameters of the fused linear layer and store them in a dictionary
    fused_linear_weight = linear_1.weight + linear_2.weight
    fused_linear_bias = linear_1.bias + linear_2.bias
    param_dict = {'weight': fused_linear_weight, 'bias': fused_linear_bias}

    # Create the fused linear layer and set its parameters
    fused_linear = nn.Linear(linear_1.in_features, linear_1.out_features)
    for name, param in fused_linear.named_parameters():
        assert param_dict.get(name) is not None
        param.data = param_dict[name]

    return fused_linear


def fuse_serial_linears(linear_1: nn.Linear, linear_2: nn.Linear) -> nn.Linear:
    """Fuse two sequential linear layers into a single linear layer."""

    # Verify the consistency of the configurations of the two linear layers
    assert linear_1.out_features == linear_2.in_features

    # Calculate the parameters of the fused linear layer and store them in a dictionary
    fused_linear_weight = linear_2.weight @ linear_1.weight
    fused_linear_bias = linear_2.bias + linear_2.weight @ linear_1.bias
    param_dict = {'weight': fused_linear_weight, 'bias': fused_linear_bias}

    # Create the fused linear layer and set its parameters
    fused_linear = nn.Linear(linear_1.in_features, linear_2.out_features)
    for name, param in fused_linear.named_parameters():
        assert param_dict.get(name) is not None
        param.data = param_dict[name]

    return fused_linear