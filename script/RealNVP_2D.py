__author__ = "Xinqiang Ding <xqding@umich.edu>"
__date__ = "2019/11/03 20:46:23"
__modified_by__ = "M.Trojanowski"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as functional


class Grassmann_Affine_Coupling(nn.Module):
    def __init__(self, n, r, hidden_dim):
        super(Grassmann_Affine_Coupling, self).__init__()
    
        self.Z = nn.Parameter(torch.randn(n-r, r) * 0.1)

        self.hidden_dim = hidden_dim
        self.n = n
        self.r = r

        ## layers used to compute scale in affine transformation
        self.scale_fc1 = nn.Linear(self.r, self.hidden_dim)
        self.scale_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.scale_fc3 = nn.Linear(self.hidden_dim, self.n)
        self.scale = nn.Parameter(torch.Tensor(self.n))
        init.normal_(self.scale)

        ## layers used to compute translation in affine transformation 
        self.translation_fc1 = nn.Linear(self.r, self.hidden_dim)
        self.translation_fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.translation_fc3 = nn.Linear(self.hidden_dim, self.n)
    
    def _compute_Y(self):
        Ir = torch.eye(self.r, device=self.Z.device)
        
        # Intermediate term: (I + Z^T Z)
        # We use solve(A, B) to get A^-1 @ B
        A = Ir + self.Z.t() @ self.Z
        
        # Calculate the top block: (I - Z^T Z) (I + Z^T Z)^-1
        # Equivalent to solving A @ Y1.T = (I - Z^T Z).T
        top_numerator = Ir - self.Z.t() @ self.Z
        Y1 = torch.linalg.solve(A, top_numerator.t()).t()
        
        # Calculate the bottom block: 2Z (I + Z^T Z)^-1
        # Equivalent to solving A @ Y2.T = (2Z).T
        Y2 = torch.linalg.solve(A, (2 * self.Z).t()).t()
        
        # Vertically stack to get n x r matrix
        return torch.cat([Y1, Y2], dim=0)

    def _compute_scale(self, c):
        ## compute scaling factor using unchanged part of x with a neural network
        s = torch.relu(self.scale_fc1(c))
        s = torch.relu(self.scale_fc2(s))
        s = torch.relu(self.scale_fc3(s)) * self.scale        
        return s

    def _compute_translation(self, c):
        ## compute translation using unchanged part of x with a neural network        
        t = torch.relu(self.translation_fc1(c))
        t = torch.relu(self.translation_fc2(t))
        t = self.translation_fc3(t)        
        return t
    
    def _to_grassmann_base(self, x, Y):
        c = x @ Y  # (batch_size, r)
        z = x - c @ Y.t()  # (batch_size, n), orthogonal part
        return c, z

    def _to_canonical_base(self, c, z, Y):
        return c @ Y.t() + z
    
    def forward(self, x):
        # get current values of orthogonal matrix Y and projection onto orthogonal complement
        Y = self._compute_Y()
        P = torch.eye(self.n, device=Y.device) - Y @ Y.t()  # Projection onto orthogonal complement, shape (n, n)

        # transform x to Grassmann base
        c, z = self._to_grassmann_base(x, Y)
        
        # apply affine transformation to orthogonal component
        s = self._compute_scale(c)
        t = self._compute_translation(c)
        z = z * torch.exp(s) + t
        z = z @ P

        # transform back to canonical base
        y = self._to_canonical_base(c, z, Y)
        
        logdet = torch.sum(s, -1) - torch.logdet(Y.t() @ (Y * torch.exp(s).unsqueeze(-1)))
        
        return y, logdet

    def inverse(self, y):
        # get current values of orthogonal matrix Y and projection onto orthogonal complement
        Y = self._compute_Y()
        P = torch.eye(self.n, device=Y.device) - Y @ Y.t()  # Projection onto orthogonal complement, shape (n, n)

        # transform y to Grassmann base
        c, z = self._to_grassmann_base(y, Y)

        # apply inverse transformation
        s = self._compute_scale(c)
        t = self._compute_translation(c)
        z = (z - t) * torch.exp(-s)
        z = z @ P

        # transform back to canonical base
        x = self._to_canonical_base(c, z, Y)

        logdet = -torch.sum(s, -1) + torch.logdet(Y.t() @ (Y * torch.exp(s).unsqueeze(-1)))
        
        return x, logdet

    
class RealNVP_2D(nn.Module):
    '''
    A vanilla RealNVP class for modeling 2 dimensional distributions
    '''
    
    def __init__(self, n, r, num_layers, hidden_dim):
        '''
        initialized with a list of masks. each mask define an affine coupling layer
        '''
        super(RealNVP_2D, self).__init__()        
        self.hidden_dim = hidden_dim 

        self.affine_couplings = nn.ModuleList(
            [Grassmann_Affine_Coupling(n, r, self.hidden_dim)
             for _ in range(num_layers)])
        
    def forward(self, x):
        ## convert latent space variables into observed variables
        y = x
        logdet_tot = 0
        for i in range(len(self.affine_couplings)):
            y, logdet = self.affine_couplings[i](y)
            logdet_tot = logdet_tot + logdet

        ## a normalization layer is added such that the observed variables is within
        ## the range of [-4, 4].
        logdet = torch.sum(torch.log(torch.abs(4*(1-(torch.tanh(y))**2))), -1)        
        y = 4*torch.tanh(y)
        logdet_tot = logdet_tot + logdet
        
        return y, logdet_tot

    def inverse(self, y):
        ## convert observed variables into latent space variables        
        x = y        
        logdet_tot = 0

        # inverse the normalization layer
        logdet = torch.sum(torch.log(torch.abs(1.0/4.0* 1/(1-(x/4)**2))), -1)
        x  = 0.5*torch.log((1+x/4)/(1-x/4))
        logdet_tot = logdet_tot + logdet

        ## inverse affine coupling layers
        for i in range(len(self.affine_couplings)-1, -1, -1):
            x, logdet = self.affine_couplings[i].inverse(x)
            logdet_tot = logdet_tot + logdet
            
        return x, logdet_tot


if __name__ == "__main__":
    # Test Grassmann_Affine_Coupling
    n = 10  # input dimension
    r = 3   # Grassmann manifold dimension
    hidden_dim = 16
    batch_size = 4
    
    # Initialize the coupling layer
    coupling = Grassmann_Affine_Coupling(n, r, hidden_dim)
    
    # Generate random input
    x = torch.randn(batch_size, n)
    
    print("Testing Grassmann_Affine_Coupling")
    print(f"Input shape: {x.shape}")
    
    # Test forward pass
    y, logdet_forward = coupling.forward(x)
    print(f"Forward pass:")
    print(f"  Output shape: {y.shape}")
    print(f"  Logdet shape: {logdet_forward.shape}")
    print(f"  Logdet values: {logdet_forward}")
    
    # Test inverse pass
    x_reconstructed, logdet_inverse = coupling.inverse(y)
    print(f"\nInverse pass:")
    print(f"  Reconstructed x shape: {x_reconstructed.shape}")
    print(f"  Logdet shape: {logdet_inverse.shape}")
    print(f"  Logdet values: {logdet_inverse}")
    
    # Check reconstruction error
    reconstruction_error = torch.norm(x - x_reconstructed).item()
    print(f"\nReconstruction error: {reconstruction_error:.6e}")
    
    # Check logdet consistency (logdet_forward + logdet_inverse should ≈ 0)
    logdet_sum = torch.mean(logdet_forward + logdet_inverse).item()
    print(f"Mean(logdet_forward + logdet_inverse): {logdet_sum:.6e}")
    
    # Test backward pass
    print(f"\nBackward pass:")
    x_grad = torch.randn(batch_size, n, requires_grad=True)
    y_grad, logdet_grad = coupling.forward(x_grad)
    loss = y_grad.sum() + logdet_grad.sum()
    loss.backward()
    
    if x_grad.grad is not None:
        print(f"  Gradient computed successfully")
        print(f"  Gradient shape: {x_grad.grad.shape}")
        print(f"  Gradient norm: {torch.norm(x_grad.grad).item():.6e}")
    else:
        print(f"  ERROR: No gradient computed!")
    
    # Test RealNVP_2D
    print("\n" + "="*60)
    print("Testing RealNVP_2D")
    print("="*60)
    
    n = 10  # input dimension
    r = 3   # Grassmann manifold dimension
    num_layers = 3  # number of coupling layers
    hidden_dim = 16
    batch_size = 4
    
    # Initialize RealNVP model
    model = RealNVP_2D(n, r, num_layers, hidden_dim)
    
    # Generate random latent input
    z = torch.randn(batch_size, n)
    
    print(f"\nLatent input shape: {z.shape}")
    
    # Test forward pass (latent -> observed)
    x, logdet_forward = model.forward(z)
    print(f"Forward pass (latent -> observed):")
    print(f"  Output shape: {x.shape}")
    print(f"  Output range: [{x.min().item():.3f}, {x.max().item():.3f}]")
    print(f"  Logdet shape: {logdet_forward.shape}")
    print(f"  Logdet values: {logdet_forward}")
    
    # Test inverse pass (observed -> latent)
    z_reconstructed, logdet_inverse = model.inverse(x)
    print(f"\nInverse pass (observed -> latent):")
    print(f"  Reconstructed z shape: {z_reconstructed.shape}")
    print(f"  Logdet shape: {logdet_inverse.shape}")
    print(f"  Logdet values: {logdet_inverse}")
    
    # Check reconstruction error
    reconstruction_error = torch.norm(z - z_reconstructed).item()
    print(f"\nReconstruction error: {reconstruction_error:.6e}")
    
    # Check logdet consistency (logdet_forward + logdet_inverse should ≈ 0)
    logdet_sum = torch.mean(logdet_forward + logdet_inverse).item()
    print(f"Mean(logdet_forward + logdet_inverse): {logdet_sum:.6e}")
    
    # Test backward pass
    print(f"\nBackward pass:")
    z_grad = torch.randn(batch_size, n, requires_grad=True)
    x_grad, logdet_grad = model.forward(z_grad)
    loss = x_grad.sum() + logdet_grad.sum()
    loss.backward()
    
    if z_grad.grad is not None:
        print(f"  Gradient computed successfully")
        print(f"  Gradient shape: {z_grad.grad.shape}")
        print(f"  Gradient norm: {torch.norm(z_grad.grad).item():.6e}")
    else:
        print(f"  ERROR: No gradient computed!")


