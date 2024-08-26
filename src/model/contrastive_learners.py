import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ContrastiveLearner(nn.Module):
    """
    A neural network model for contrastive learning using a specified encoder and a projection head.

    Attributes:
        encoder (nn.Module): The backbone encoder network.
        projection_head (nn.Sequential): The projection head consisting of fully connected layers.
    """
    def __init__(self, encoder_name='resnet18', projection_dim=128):
        """
        Initializes the ContrastiveLearner with a specified encoder and projection dimension.

        Args:
            encoder_name (str): The name of the encoder model to use (default is 'resnet18').
            projection_dim (int): The dimension of the projection space (default is 128).

        References:
            - "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
              https://arxiv.org/abs/2002.05709
        """
        super(ContrastiveLearner, self).__init__()
        self.encoder = self._get_encoder(encoder_name)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 512), # TODO: do this automatically
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def _get_encoder(self, encoder_name):
        """
        Retrieves the specified encoder model and removes its classification layer.

        Args:
            encoder_name (str): The name of the encoder model to use.

        Returns:
            nn.Module: The encoder model without the classification layer.
        """
        if (encoder_name == 'resnet18'):
            encoder = models.resnet18(pretrained=False) # TODO: experiment with either pretrained true or false.
            self.encoder_output_dim = encoder.fc.in_features
            encoder = nn.Sequential(*list(encoder.children())[:-1])
        elif encoder_name == 'mobilenet_v2':
            encoder = models.mobilenet_v2(weights=False)
            self.encoder_output_dim = encoder.classifier[1].in_features
            encoder = nn.Sequential(*list(encoder.children())[:-1])  # Remove the classification head
        else:
            raise ValueError(f"Encoder {encoder_name} is not supported")
        return encoder

    def forward(self, x):
        """
        Forward pass through the encoder and projection head, followed by L2 normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized projections.
        """
        features = self.encoder(x)
        features = self.pool(features)
        
        try:
            features = features.view(features.size(0), -1)  # Flatten from [batch_size, channels, H, W] to [batch_size, channels*H*W]
        except Exception as e:
            raise ValueError(f"Features have incorrect dimensions: {e}")
        # Check if features have a single channel and remove the extra dimension, if present
        if features.dim() == 4 and features.size(2) == 1 and features.size(3) == 1:
            features = features.view(features.size(0), -1)

        projections = self.projection_head(features)
        projections = F.normalize(projections, dim=1)
        return projections

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss for contrastive learning.

    Attributes:
        batch_size (int): The batch size.
        temperature (float): The temperature scaling factor.
        mask (torch.Tensor): The mask to exclude positive samples from the negative samples.
        criterion (nn.CrossEntropyLoss): The cross-entropy loss function.
    """
    def __init__(self, batch_size, temperature=0.5, device='cpu'):
        """
        Initializes the NTXentLoss with a specified batch size and temperature.

        Args:
            batch_size (int): The batch size.
            temperature (float): The temperature scaling factor (default is 0.5).

        References:
            - "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
              https://arxiv.org/abs/2002.05709
            - "Momentum Contrast for Unsupervised Visual Representation Learning" (MoCo)
              https://arxiv.org/pdf/1911.05722
            - "Representation Learning with Contrastive Predictive Coding"
              https://arxiv.org/abs/1807.03748
        """
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self._get_correlated_mask().to(device)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def _get_correlated_mask(self):
        """
        Creates a mask to exclude positive samples from the negative samples.

        Returns:
            torch.Tensor: The mask tensor.
        """
        mask = torch.ones((self.batch_size * 2, self.batch_size * 2), dtype=bool).to(self.device)
        mask.fill_diagonal_(0)
        for i in range(self.batch_size):
            mask[i, self.batch_size + i] = 0
            mask[self.batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        Computes the NTXentLoss between two sets of projections.

        Args:
            z_i (torch.Tensor): Projections from the first set of inputs.
            z_j (torch.Tensor): Projections from the second set of inputs.

        Returns:
            torch.Tensor: The computed NTXentLoss.
        """
        batch_size = z_i.size(0)  # Dynamically get batch size for the current batch
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        # Create the similarity matrices
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        # Adjust the reshaping step based on the actual batch size
        positive_samples = torch.cat((sim_i_j, sim_j_i)).view(-1, 1)

        # Recreate mask dynamically based on the current batch size
        mask = torch.ones((batch_size * 2, batch_size * 2), dtype=bool).to(z.device)
        mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        negative_samples = sim[mask].view(batch_size * 2, -1)
        labels = torch.zeros(batch_size * 2).to(torch.long).to(z.device)
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        return loss / (2 * batch_size)

if __name__ == "__main__":
    """
    Example usage of the ContrastiveLearner and NTXentLoss classes.
    """
    model = ContrastiveLearner(encoder_name='resnet18')
    criterion = NTXentLoss(batch_size=32)
    
    x1 = torch.randn(32, 3, 224, 224)
    x2 = torch.randn(32, 3, 224, 224)
    
    z_i = model(x1)
    z_j = model(x2)
    
    loss = criterion(z_i, z_j)
    print(f"Contrastive Loss: {loss.item()}")