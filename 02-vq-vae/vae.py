# Borrow Code from https://github.com/AntixK/PyTorch-VAE/blob/a6896b944c918dd7030e7d795a8c13e5c6345ec7/models/vanilla_vae.py 
from typing import List, Tuple

import torch 
from torch import nn, Tensor
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self,
                 in_channels: int,
                 input_size: Tuple,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VAE, self).__init__()
        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.hidden_dims = hidden_dims.copy() 

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        hidden_dim_size = len(hidden_dims)
        feat_map_size = int(input_size[0]/(2**hidden_dim_size)) * int(input_size[1]/(2**hidden_dim_size))
        self.encoder = nn.Sequential(*modules)
        print(f'hidden_dims[-1]*4, latent_dim : {hidden_dims[-1]*feat_map_size, latent_dim}')
        self.fc_mu = nn.Linear(hidden_dims[-1]*feat_map_size, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*feat_map_size, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * feat_map_size)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        print(F'input {input.shape}')
        result = self.encoder(input)
        self.feat_map_size = result.shape[2:]
        print(F'result {result.shape}')
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        print(f'self.feat_map_size : {self.feat_map_size}')
        print(f'self.hidden_dims[-1] : {self.hidden_dims[-1]} {self.hidden_dims}')
        result = result.view(-1, self.hidden_dims[-1], self.feat_map_size[0], self.feat_map_size[1])
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        print(f'mu {mu.shape} log_var {log_var.shape}')
        z = self.reparameterize(mu, log_var)
        # return  [self.decode(z), input, mu, log_var]
        return self.decode(z)

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

def main():
    x = torch.rand(1, 3, 64, 64)
    print(x.shape)
    model = VAE(in_channels=3, input_size=(64,64), latent_dim=2048, hidden_dims=[32, 64, 128, 256, 512])
    if torch.cuda.is_available():
      x, model = x.to("cuda:0"), model.to("cuda:0")
    out = model(x)
    print(f'out {len(out)} {type(out)} {out.shape}')
    
    x2 = torch.rand(1, 3, 128, 128)
    print(x2.shape)
    model2 = VAE(in_channels=3, input_size=(128, 128), latent_dim=2048, hidden_dims=[32, 64, 128, 256, 512])
    if torch.cuda.is_available():
      x2, model2 = x2.to("cuda:0"), model2.to("cuda:0")
    out2 = model2(x2)
    print(f'out2 {len(out2)} {type(out2)} {out2.shape}')

if __name__=="__main__":
    main()
