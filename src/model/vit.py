import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights

# import ViT_B_16_Weights.IMAGENET1K_V1


class SmallFormer(nn.Module):
    """
    A small Vision Transformer (ViT) model for image classification.

    This model consists of a Vision Transformer backbone followed by a multi-layer perceptron (MLP)
    for classification. It's designed to be a lightweight version of larger ViT architectures.

    Args:
        num_classes (int): The number of output classes. Default is 1 for binary classification.

    Attributes:
        vit (nn.Module): The Vision Transformer backbone (vit_b_16) without pre-trained weights.
        mlp (nn.Sequential): A multi-layer perceptron for feature processing and classification.
        sigmoid (nn.Sigmoid): Sigmoid activation for the final output.

    Forward Pass:
        x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
        Returns: torch.Tensor of shape (batch_size, num_classes)

    Note:
        The sigmoid activation at the end may introduce stability but might not be necessary
        for all use cases. Consider removing it for multi-class classification tasks.
    """

    def __init__(self, num_classes: int = 1):
        super(SmallFormer, self).__init__()
        self.vit = vit_b_16(weights=None)
        self.mlp = nn.Sequential(
            nn.Linear(1000, 512), nn.SiLU(), nn.BatchNorm1d(512), nn.Linear(512, 84), nn.SiLU(), nn.BatchNorm1d(84), nn.Linear(84, num_classes)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.vit(x)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x


class LayerNorm(nn.Module):
    """
    A custom implementation of Layer Normalization.

    This module applies Layer Normalization over a mini-batch of inputs as described
    in the paper "Layer Normalization" (https://arxiv.org/abs/1607.06450).

    Args:
        normalized_shape (int or tuple): The shape of the input tensor or the size of the last dimension.
        eps (float): A value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): If True, this module has learnable per-element affine parameters. Default: True

    Shape:
        - Input: (*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1])
        - Output: (*, normalized_shape[0], normalized_shape[1], ..., normalized_shape[-1])

    Attributes:
        weight (Tensor): The learnable weights of the module of shape `normalized_shape`
                         if `elementwise_affine` is True.
        bias (Tensor): The learnable bias of the module of shape `normalized_shape`
                       if `elementwise_affine` is True.

    Examples::
        >>> m = LayerNorm(100)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)
    """

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / (std + self.eps)
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        return x


class MLPwithSkipConnection(nn.Module):
    """
    Multi-Layer Perceptron with Skip Connection.

    This module implements a single MLP layer with a skip connection.
    It applies a linear transformation followed by layer normalization,
    SiLU activation, and dropout. The output is then added to the input
    (after optional dimension adjustment).

    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        dropout (float, optional): Dropout rate. Default: 0.1.

    Attributes:
        mlp (nn.Sequential): The main MLP path including linear layer,
                             layer normalization, activation, and dropout.
        skip (nn.Module): Skip connection, either identity or linear projection.

    Shape:
        - Input: (N, *, input_dim)
        - Output: (N, *, output_dim)
        where N is the batch size and * is any number of additional dimensions.

    Examples:
        >>> m = MLPwithSkipConnection(100, 100)
        >>> input = torch.randn(20, 100)
        >>> output = m(input)
    """

    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super(MLPwithSkipConnection, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            # nn.BatchNorm1d(output_dim), # TODO investigate when to use batchnorm and when to use layernorm
            LayerNorm(output_dim, elementwise_affine=True),  # TODO: experiment with elementwise_affine on and off
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.skip = (
            nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        )  # linear transformation if input_dim != output_dim, else identity

    def forward(self, x) -> torch.Tensor:
        x = self.mlp(x) + self.skip(x)
        return x


class LargeFormer(nn.Module):
    """
    LargeFormer: A Vision Transformer (ViT) based model for image classification.

    This class implements a large-scale Vision Transformer model, utilizing a pre-trained
    ViT-B/16 as the backbone, followed by a custom MLP head for classification.

    Args:
        num_classes (int): Number of output classes. Default is 1 for binary classification.
        pretrained_weights (str): Specifies the pre-trained weights to use.
                                  If None, random initialization is used.

    Attributes:
        vit (nn.Module): The Vision Transformer backbone (ViT-B/16).
        vit_out_shape (int): The output dimension of the ViT backbone (768 for ViT-B/16).
        mlp (nn.Sequential): Custom MLP head for classification.
        sigmoid (nn.Sigmoid): Sigmoid activation for binary classification.

    Forward shape:
        - Input: (N, C, H, W)
        - Output: (N, num_classes)
        where N is the batch size, C is the number of channels,
        H and W are the height and width of the input image.

    Example:
        >>> model = LargeFormer(num_classes=1, pretrained_weights="IMAGENET1K_V1")
        >>> input = torch.randn(1, 3, 224, 224)
        >>> output = model(input)
    """

    def __init__(self, num_classes: int = 1, pretrained_weights: str = None):
        super(LargeFormer, self).__init__()
        self.vit = vit_b_16(weights=pretrained_weights)  # "IMAGENET1K_V1" if use_pretrained else None)
        self.vit.heads = nn.Identity()  # remove clf head, as we will add our own
        # get out shape of vit
        self.vit_out_shape = 768

        self.mlp = nn.Sequential(
            MLPwithSkipConnection(self.vit_out_shape, 512), MLPwithSkipConnection(512, 84), MLPwithSkipConnection(84, 16), nn.Linear(16, num_classes)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        x = self.vit(x)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x


class MegaFormer(nn.Module):
    """
    MegaFormer: An advanced Vision Transformer (ViT) based model for image classification.

    This class implements a large-scale Vision Transformer model, utilizing a pre-trained
    ViT-B/16 as the backbone, followed by a more complex MLP head for classification.

    Args:
        num_classes (int): Number of output classes. Default is 1 for binary classification.
        dropout (float): Dropout rate for the MLP layers. Default is 0.1.
        pretrained_weights (str): Specifies the pre-trained weights to use for the ViT backbone.
                                  If None, random initialization is used.

    Attributes:
        vit (nn.Module): The Vision Transformer backbone (ViT-B/16).
        mlp (nn.Sequential): Complex MLP head for classification with skip connections.
        sigmoid (nn.Sigmoid): Sigmoid activation for binary classification.

    Forward shape:
        - Input: (N, C, H, W)
        - Output: (N, num_classes)
        where N is the batch size, C is the number of channels,
        H and W are the height and width of the input image.

    Example:
        >>> model = MegaFormer(num_classes=1, dropout=0.1, pretrained_weights="IMAGENET1K_V1")
        >>> input = torch.randn(1, 3, 224, 224)
        >>> output = model(input)
    """

    def __init__(self, num_classes: int = 1, dropout: float = 0.1, pretrained_weights: str = None):
        super(MegaFormer, self).__init__()
        self.vit = vit_b_16(weights=pretrained_weights)
        self.vit.heads = nn.Identity()  # remove clf head, as we will add our own
        self.mlp = nn.Sequential(
            MLPwithSkipConnection(1000, 800, dropout),
            MLPwithSkipConnection(800, 400, dropout),
            MLPwithSkipConnection(400, 200, dropout),
            MLPwithSkipConnection(200, 100, dropout),
            MLPwithSkipConnection(100, 16, dropout),
            nn.Linear(16, num_classes),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x) -> torch.Tensor:
        x = self.vit(x)
        x = self.mlp(x)
        x = self.sigmoid(x)
        return x
