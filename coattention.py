import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttention(nn.Module):

    """This is the class for Hyperbolic Fourier-coattention mechanism."""
    
    def __init__(self, latent_dim = 100,  embedding_dim = 100,  fourier= False):
        super(CoAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.k = 128
        self.Wl = nn.Parameter(torch.Tensor((self.latent_dim, self.latent_dim)))
        self.Wc = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.Ws = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.whs = nn.Parameter(torch.Tensor((1, self.k)))
        self.whc = nn.Parameter(torch.Tensor((1, self.k)))
        self.concat_m1 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_m2 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_b = nn.Parameter(torch.Tensor((1, self.embedding_dim)))

        #register weights and bias as params
        self.register_parameter("Wl", self.Wl)
        self.register_parameter("Wc", self.Wc)
        self.register_parameter("Ws", self.Ws)
        self.register_parameter("whs", self.whs)
        self.register_parameter("whc", self.whc)

        #concatenation operation for hyperbolic 
        self.register_parameter("concat_m1", self.concat_m1)
        self.register_parameter("concat_m2", self.concat_m2)
        self.register_parameter("concat_b", self.concat_b)

        #initialize data of parameters
        self.Wl.data = torch.randn((self.latent_dim, self.latent_dim))
        self.Wc.data = torch.randn((self.k, self.latent_dim))
        self.Ws.data = torch.randn((self.k, self.latent_dim))
        self.whs.data = torch.randn((1, self.k))
        self.whc.data = torch.randn((1, self.k))
        self.concat_m1.data = torch.randn((1, 1))
        self.concat_m2.data = torch.randn((1, 1))
        self.concat_b.data = torch.randn((1, self.embedding_dim))
        self.c = combined_curvature
        self.fourier = fourier

    def forward(self, img_rep, cap_rep):

        """This function will return the shape [batch_size, embedding_dim]."""

        mobius_matvec = self.manifold.mobius_matvec
        proj = self.manifold.proj
        logmap0 = self.manifold.logmap0
        expmap0 = self.manifold.expmap0
        mobius_add = self.manifold.mobius_add
        curv = self.c 

        if self.fourier:
            # KFU
            img_rep = torch.fft.fft2(img_rep).float()
            cap_rep = torch.fft.fft2(cap_rep).float()


        img_rep_trans = img_rep.transpose(2, 1)#[32, 100, 50]
        cap_rep_trans = cap_rep.transpose(2, 1)#[32, 100, 10]


        L = torch.tanh(torch.matmul(torch.matmul(cap_rep, self.Wl), img_rep_trans))  
        Hs = torch.tanh(torch.matmul(self.Ws, img_rep_trans) + torch.matmul(torch.matmul(self.Wc, cap_rep_trans), L))
        Hc = torch.tanh(torch.matmul(self.Wc, cap_rep_trans)+ torch.matmul(torch.matmul(self.Ws, img_rep_trans), L.T))
        As = F.softmax(torch.matmul(self.whs, Hs), dim=2)
        Ac = F.softmax(torch.matmul(self.whc, Hc), dim=2)
        co_s = torch.matmul(As,img_rep) # (1, 100)
        co_c = torch.matmul(Ac, cap_rep) # (1, 100)
        co_sc = torch.cat([co_s, co_c], dim = -1)
        co_sc = torch.squeeze(co_sc)

        assert not torch.isnan(co_sc).any(), "co_sc is nan"
        return co_sc, As, Ac # [32, 200], 