import torch
import torch.nn as nn
import torch.nn.functional as F


class BLT_net(nn.Module):
    def __init__(self, n_blocks, n_layers, image_size=64, n_classes=100, is_lateral_enabled=True, is_topdown_enabled=True,
                 LT_interaction='additive', timesteps=10, in_channels=3, n_start_filters=32, kernel_size=3):
        super().__init__()
        self.timesteps = timesteps
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.is_lateral_enabled = is_lateral_enabled
        self.is_topdown_enabled = is_topdown_enabled

        # Initialize blocks
        self.blocks = nn.ModuleList()

        for block_idx in range(self.n_blocks):
            block = nn.ModuleList()
            out_channels = n_start_filters * (2 ** block_idx)

            for layer_idx in range(self.n_layers):
                # First layer of each block ramps up the number of channel
                if layer_idx == 0:
                    layer = BLT_Conv(in_channels, out_channels, kernel_size,
                                     is_lateral_enabled, is_topdown_enabled, LT_interaction)
                # Rest of the layers in the block have the same number of channels
                else:
                    layer = BLT_Conv(out_channels, out_channels, kernel_size,
                                     is_lateral_enabled, is_topdown_enabled, LT_interaction)

                layer_ln = nn.LayerNorm([out_channels, image_size // (2 ** block_idx), image_size // (2 ** block_idx)])
                block.append(LayerWithNorm(layer, layer_ln))

                # Input channels for the next layer is the output channels of the current layer
                in_channels = out_channels

            self.blocks.append(block)

        # Define the readout layer
        self.readout = nn.Linear(n_start_filters * 2
                                 ** (n_blocks - 1), n_classes)

    def forward(self, inputs):
        last_layer_activations = [None for _ in range(self.n_blocks)]
        outputs = [None for _ in range(self.timesteps)]

        # Repeat for each timestep
        for t in range(self.timesteps):
            x = inputs
            for block_idx, block in enumerate(self.blocks):
                for layer_idx, layer_with_norm in enumerate(block):

                    # Determine the inputs for lateral and top-down connections
                    is_last_layer = layer_idx == self.n_layers - 1
                    is_last_block = block_idx == self.n_blocks - 1
                    l_input = last_layer_activations[block_idx] if self.is_lateral_enabled and is_last_layer else None
                    t_input = last_layer_activations[block_idx + 1] if self.is_topdown_enabled and is_last_layer and not is_last_block else None

                    x = torch.relu(layer_with_norm(x, l_input, t_input))

                # Store the output of the last layer of each block for each timestep
                last_layer_activations[block_idx] = x

                # Apply max pooling after each block (except the last block)
                if block_idx < self.n_blocks - 1:
                    x = F.max_pool2d(x, kernel_size=2, stride=2)
                # Apply average pooling if it's the last block
                elif block_idx == self.n_blocks - 1:
                    # Get the height and width of the tensor
                    height, width = x.size()[2:]
                    x = F.avg_pool2d(x, (height, width))

            x = torch.flatten(x, 1)
            # Apply the readout layer at the end of each timestep
            outputs[t] = torch.log(torch.clamp(
                torch.softmax(self.readout(x), dim=1),
                1e-10, 1.0
            ))
        return outputs


class LayerWithNorm(nn.Module):
    def __init__(self, layer, layer_norm):
        super().__init__()
        self.layer = layer
        self.layer_norm = layer_norm

    def forward(self, x, l_input=None, t_input=None):
        x = self.layer(x, l_input, t_input)
        return self.layer_norm(x)


class BLT_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, is_lateral_enabled, is_topdown_enabled, LT_interaction):
        super().__init__()
        self.LT_interaction = LT_interaction
        self.bottom_up = nn.Conv2d(in_channels, out_channels, kernel_size, padding="same")

        if is_lateral_enabled:
            self.lateral = nn.Conv2d(out_channels, out_channels, kernel_size, padding="same")

        if is_topdown_enabled:
            ct_padding = int((kernel_size - 1) / 2)
            self.top_down = nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size, stride=2, padding=ct_padding, output_padding=1)

    def forward(self, b_input, l_input=None, t_input=None):
        b_input = self.bottom_up(b_input)
        l_input = self.lateral(l_input) if l_input is not None else None
        t_input = self.top_down(t_input) if t_input is not None else None

        if self.LT_interaction == "additive":
            return self.additive_combination(b_input, l_input, t_input)
        elif self.LT_interaction == "multiplicative":
            return self.multiplicative_combination(b_input, l_input, t_input)
        else:
            raise ValueError("LT_interaction must be additive or multiplicative")

    def additive_combination(self, b_input, l_input, t_input):
        result = b_input
        if l_input is not None:
            result += l_input
        if t_input is not None:
            result += t_input
        return result

    def multiplicative_combination(self, b_input, l_input, t_input):
        if l_input is not None and t_input is not None:
            return b_input * (1.0 + l_input + t_input)
        if l_input is not None:
            return b_input * (1.0 + l_input)
        if t_input is not None:
            return b_input * (1.0 + t_input)
        return b_input