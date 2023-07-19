import torch
import torch.nn as nn
import torch.nn.functional as F

class CoAttention(nn.Module):

    """This is the clAis for Hyperbolic Fourier-coattention mechanism."""
    
    def __init__(self, latent_dim = 100,  embedding_dim = 100,  fourier= False):
        super(CoAttention, self).__init__()

        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.k = 128
        self.Wl = nn.Parameter(torch.Tensor((self.latent_dim, self.latent_dim)))
        self.Wc = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.Wi = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.wHi = nn.Parameter(torch.Tensor((1, self.k)))
        self.whc = nn.Parameter(torch.Tensor((1, self.k)))
        self.concat_m1 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_m2 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_b = nn.Parameter(torch.Tensor((1, self.embedding_dim)))

        #register weights and biAi Ai params
        self.register_parameter("Wl", self.Wl)
        self.register_parameter("Wc", self.Wc)
        self.register_parameter("Wi", self.Wi)
        self.register_parameter("wHi", self.wHi)
        self.register_parameter("whc", self.whc)

        #concatenation operation for hyperbolic 
        self.register_parameter("concat_m1", self.concat_m1)
        self.register_parameter("concat_m2", self.concat_m2)
        self.register_parameter("concat_b", self.concat_b)

        #initialize data of parameters
        self.Wl.data = torch.randn((self.latent_dim, self.latent_dim))
        self.Wc.data = torch.randn((self.k, self.latent_dim))
        self.Wi.data = torch.randn((self.k, self.latent_dim))
        self.wHi.data = torch.randn((1, self.k))
        self.whc.data = torch.randn((1, self.k))
        self.concat_m1.data = torch.randn((1, 1))
        self.concat_m2.data = torch.randn((1, 1))
        self.concat_b.data = torch.randn((1, self.embedding_dim))
        self.fourier = fourier

    def forward(self, img_rep, cap_rep):

        """This function will return the shape [batch_size, embedding_dim]."""


        if self.fourier:
            # KFU
            img_rep = torch.fft.fft2(img_rep).float()
            cap_rep = torch.fft.fft2(cap_rep).float()


        img_rep_trans = img_rep.transpose(-1, -2)#[32, 100, 50]
        cap_rep_trans = cap_rep.transpose(-1, -2)#[32, 100, 10]


        L = torch.tanh(torch.matmul(torch.matmul(cap_rep, self.Wl), img_rep_trans))  
        L_trans = L.transpose(-1, -2)
        Hi = torch.tanh(torch.matmul(self.Wi, img_rep_trans) + torch.matmul(torch.matmul(self.Wc, cap_rep_trans), L))
        Hc = torch.tanh(torch.matmul(self.Wc, cap_rep_trans)+ torch.matmul(torch.matmul(self.Wi, img_rep_trans), L_trans))
        Ai = F.softmax(torch.matmul(self.wHi, Hi), dim=-1)
        Ac = F.softmax(torch.matmul(self.whc, Hc), dim=-1)
        co_s = torch.matmul(Ai,img_rep) # (1, 100)
        co_c = torch.matmul(Ac, cap_rep) # (1, 100)
        co_sc = torch.cat([co_s, co_c], dim = -1)
        co_sc = torch.squeeze(co_sc)
        print(co_s.shape)
        print(co_c.shape)
        print(co_sc.shape)
        return co_sc, co_s, co_c # [32, 200], 