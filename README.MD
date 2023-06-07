# BLT Net

This repository contains the implementation of the BLT Net.
The model is implemented in PyTorch and PyTorch Lightning and provides customisation options for various number of blocks, layers, recurrent interaction types, and timesteps.

## Dependecies

Please make sure https://github.com/KietzmannLab/pytorch-dataset-loaders and it's dependencies are installed.

## Usage

First, import the BLT Net model:
  
```python
from blt_net.blt_net import BLT_net
```

Then, you can create an instance of the model and use it:
```python
model = BLT_net(
    n_blocks=5,
    n_layers=1,
    is_lateral_enabled=True,
    is_topdown_enabled=True,
    LT_interaction="additive",
    timesteps=10,
)
```

Here is a breakdown of the parameters used to initialise the model:

**n_blocks**: Number of convolutional blocks in the network. Each block is composed of several layers. This parameter allows you to control the depth and capacity of the model.

**n_layers**: Number of layers in each block. Allows control over the complexity within each block.

**is_lateral_enabled**: Enables or disables lateral connections in the model. Lateral connections are those that connect a layer to its corresponding layer in the next timestep. They can help propagate information and gradients across timesteps.

**is_topdown_enabled**: Enables or disables top-down connections. These are connections that go from a block to the block below it in the next timestep. They can provide a form of feedback or context to the lower layers.

**LT_interaction**: Determines how information from the bottom-up, lateral, and top-down connections are combined. If set to "additive", the information is simply added together. If set to "multiplicative", the information is combined in a multiplicative manner.

**timesteps**: Determines the number of timesteps the network should process. This can be considered as a form of 'unrolling' the network over time. A higher number of timesteps allows more information to be propagated through time, but at the cost of increased computation.

## Contributing
Contributions are very welcome. If you know how to improve this model, feel free to fork this repository, make your changes and submit a pull request.