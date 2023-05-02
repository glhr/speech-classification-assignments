import torch

# given an input size and kernel parameters, calculate the output size for a single layer
def get_output_size_single_layer(input_size=(1,101,40), out_channels=None, kernel_size=(5,5), stride=(1,1), padding=(0,0)):
    Din,Hin,Win = input_size # input depth, height, width
    Hout = (Hin - kernel_size[0] + 2*padding[0])//stride[0] + 1 # output height
    Wout = (Win - kernel_size[1] + 2*padding[1])//stride[1] + 1 # output width
    Dout = out_channels if out_channels is not None else Din # if no output_channels specified, assume same as input_channels (e.g. for pooling)
    return (Dout,Hout,Wout)



# loop through the layers, passing the output size from one layer to the next
def get_output_size_from_layer_params(input_size=(1,101,40), layer_params={"layer1": {"out_channels": 32, "kernel_size": (5,5), "stride": (1,1), "padding": (0,0)}}):
    output_size = input_size
    for layer in layer_params:
        output_size = get_output_size_single_layer(input_size=output_size, **layer_params[layer])
        print(f"Output size after {layer}: {output_size}")

    flattened_size = torch.LongTensor(output_size).prod()
    print(f"Final output size: {output_size} -> when flattened {flattened_size}")
    return flattened_size



if __name__ == "__main__":
    # define the layer parameters for our model
    layer_params = {
        "conv1": {"out_channels": 32, "kernel_size": (5,5), "stride": (1,1), "padding": (0,0)},
        "pool1": {"kernel_size": (2,2), "stride": (2,2), "padding": (0,0)},
        "conv2": {"out_channels": 16, "kernel_size": (5,5), "stride": (1,1), "padding": (0,0)},
        "pool2": {"kernel_size": (2,2), "stride": (2,2), "padding": (0,0)},
    }

    get_output_size_from_layer_params(input_size=(1,101,40), layer_params=layer_params)