import torch.nn
from typing import Tuple, Optional

import geoopt
from geoopt import Manifold
from geoopt import Lorentz as LorentzOri
from geoopt.utils import size2shape
from .lmath import * 
from .utils import acosh


def arcosh(x: torch.Tensor):
    dtype = x.dtype
    z = torch.sqrt(torch.clamp_min(x.pow(2) - 1.0, 1e-7))
    return torch.log(x + z).to(dtype)


class Lorentz(LorentzOri):
    def __init__(self, k=1.0, learnable=False):
        super().__init__(k, learnable)

    def _check_point_on_manifold(
        self, x: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        dn = x.size(dim) - 1
        x = x ** 2
        quad_form = -x.narrow(dim, 0, 1) + x.narrow(dim, 1, dn).sum(
            dim=dim, keepdim=True
        )
        ok = torch.allclose(quad_form, -self.k, atol=atol, rtol=rtol)
        if not ok:
            reason = f"'x' minkowski quadratic form is not equal to {-self.k.item()}"
        else:
            reason = None
        return ok, reason

    def _check_vector_on_tangent(
        self, x: torch.Tensor, u: torch.Tensor, *, atol=1e-5, rtol=1e-5, dim=-1
    ) -> Tuple[bool, Optional[str]]:
        inner_ = inner(u, x, dim=dim)
        ok = torch.allclose(inner_, torch.zeros(1), atol=atol, rtol=rtol)
        if not ok:
            reason = "Minkowski inner produt is not equal to zero"
        else:
            reason = None
        return ok, reason

    def dist(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        return dist(x, y, k=self.k, keepdim=keepdim, dim=dim)
    
    def dist_batch(
        self, x: torch.Tensor, y: torch.Tensor, *, keepdim=False, dim=-1
    ) -> torch.Tensor:
        out = []
        for i in range(x.shape[0]):
            out.append(self.dist(x[i], y).unsqueeze_(0))
        return  torch.cat(out,dim=0)

    

    def dist0(self, x: torch.Tensor, *, dim=-1, keepdim=False) -> torch.Tensor:
        return dist0(x, k=self.k, dim=dim, keepdim=keepdim)

    def cdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x = x.clone()
        # x.narrow(-1, 0, 1).mul_(-1)
        # return torch.sqrt(self.k) * acosh(-(x.matmul(y.transpose(-1, -2))) / self.k)
        return cdist(x, y, k=self.k)
        # return -(x.matmul(y.transpose(-1, -2))) / self.k

    def sqdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -2 - 2 * inner(x, y)

    def lorentz_to_klein(self, x):
        dim = x.shape[-1] - 1
        return acosh(x.narrow(-1, 1, dim) / x.narrow(-1, 0, 1))

    def klein_to_lorentz(self, x):
        norm = (x * x).sum(dim=-1, keepdim=True)
        size = x.shape[:-1] + (1, )
        return torch.cat([x.new_ones(size), x], dim=-1) / torch.clamp_min(torch.sqrt(1 - norm), 1e-7)

    def lorentz_to_poincare(self, x):
        return lorentz_to_poincare(x, self.k)

    def norm(self, u: torch.Tensor, *, keepdim=False, dim=-1) -> torch.Tensor:
        return norm(u, keepdim=keepdim, dim=dim)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return egrad2rgrad(x, u, dim=dim)

    def projx(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return project(x, k=self.k, dim=dim)

    def proju(self, x: torch.Tensor, v: torch.Tensor, *, dim=-1) -> torch.Tensor:
        v = project_u(x, v, k=self.k, dim=dim)
        return v

    def proju0(self, v: torch.Tensor) -> torch.Tensor:
        v = project_u0(v)
        return v

    def expmap(
        self, x: torch.Tensor, u: torch.Tensor, *, norm_tan=True, proj=True, dim=-1
    ) -> torch.Tensor:
        if norm_tan is True:
            u = self.proju(x, u, dim=dim)
        res = expmap(x, u, k=self.k, dim=dim)
        if proj is True:
            return project(res, k=self.k, dim=dim)
        else:
            return res

    def expmap0(self, u: torch.Tensor, *, proj=True, dim=-1) -> torch.Tensor:
        res = expmap0(u, k=self.k, dim=dim)
        if proj:
            return project(res, k=self.k, dim=dim)
        else:
            return res

    def logmap(self, x: torch.Tensor, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return logmap(x, y, k=self.k, dim=dim)

    def clogmap(self, x, y):
        return clogmap(x, y)

    def logmap0(self, y: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return logmap0(y, k=self.k, dim=dim)

    def logmap0back(self, x: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return logmap0back(x, k=self.k, dim=dim)

    def inner(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor = None,
        *,
        keepdim=False,
        dim=-1,
    ) -> torch.Tensor:
        # TODO: x argument for maintaining the support of optims
        if v is None:
            v = u
        return inner(u, v, dim=dim, keepdim=keepdim)

    def inner0(self, v: torch.Tensor = None, *, keepdim=False, dim=-1,) -> torch.Tensor:
        return inner0(v, k=self.k, dim=dim, keepdim=keepdim)

    def cinner(self, x: torch.Tensor, y: torch.Tensor):
        # x = x.clone()
        # x.narrow(-1, 0, 1).mul_(-1)
        # return x @ y.transpose(-1, -2)
        return cinner(x, y)

    def egrad2rgrad(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return egrad2rgrad(x, u, k=self.k, dim=dim)

    def transp(
        self, x: torch.Tensor, y: torch.Tensor, v: torch.Tensor, *, dim=-1
    ) -> torch.Tensor:
        return parallel_transport(x, y, v, k=self.k, dim=dim)

    def transp0(self, y: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return parallel_transport0(y, u, k=self.k, dim=dim)

    def transp0back(self, x: torch.Tensor, u: torch.Tensor, *, dim=-1) -> torch.Tensor:
        return parallel_transport0back(x, u, k=self.k, dim=dim)

    def transp_follow_expmap(
        self, x: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, dim=-1, proj=True
    ) -> torch.Tensor:
        y = self.expmap(x, u, dim=dim, proj=proj)
        return self.transp(x, y, v, dim=dim)

    def mobius_add(self, x, y):
        v = self.logmap0(y)
        v = self.transp0(x, v)
        return self.expmap(x, v)

    def geodesic_unit(
        self, t: torch.Tensor, x: torch.Tensor, u: torch.Tensor, *, dim=-1, proj=True
    ) -> torch.Tensor:
        res = geodesic_unit(t, x, u, k=self.k)
        if proj:
            return project(res, k=self.k, dim=dim)
        else:
            return res

    def random_normal(
        self, *size, mean=0, std=1, dtype=None, device=None
    ) -> "geoopt.ManifoldTensor":
        r"""
        Create a point on the manifold, measure is induced by Normal distribution on the tangent space of zero.

        Parameters
        ----------
        size : shape
            the desired shape
        mean : float|tensor
            mean value for the Normal distribution
        std : float|tensor
            std value for the Normal distribution
        dtype: torch.dtype
            target dtype for sample, if not None, should match Manifold dtype
        device: torch.device
            target device for sample, if not None, should match Manifold device

        Returns
        -------
        ManifoldTensor
            random points on Hyperboloid

        Notes
        -----
        The device and dtype will match the device and dtype of the Manifold
        """
        self._assert_check_shape(size2shape(*size), "x")
        if device is not None and device != self.k.device:
            raise ValueError(
                "`device` does not match the projector `device`, set the `device` argument to None"
            )
        if dtype is not None and dtype != self.k.dtype:
            raise ValueError(
                "`dtype` does not match the projector `dtype`, set the `dtype` arguement to None"
            )
        tens = torch.randn(*size, device=self.k.device, dtype=self.k.dtype) * std + mean
        tens /= tens.norm(dim=-1, keepdim=True)
        return geoopt.ManifoldTensor(self.expmap0(tens), manifold=self)

    def origin(
        self, *size, dtype=None, device=None, seed=42
    ) -> "geoopt.ManifoldTensor":
        """
        Zero point origin.

        Parameters
        ----------
        size : shape
            the desired shape
        device : torch.device
            the desired device
        dtype : torch.dtype
            the desired dtype
        seed : int
            ignored

        Returns
        -------
        ManifoldTensor
            zero point on the manifold
        """
        if dtype is None:
            dtype = self.k.dtype
        if device is None:
            device = self.k.device

        zero_point = torch.zeros(*size, dtype=dtype, device=device)
        zero_point[..., 0] = torch.sqrt(self.k)
        return geoopt.ManifoldTensor(zero_point, manifold=self)

    def mid_point(self, x, w=None):
        if w is not None:
            ave = w.matmul(x)
        else:
            ave = x.mean(dim=-2)
        denom = (-self.inner(ave, ave, keepdim=True))
        # mask = denom < 0
        # denom[mask].clamp_max_(-1e-8)
        # denom[~mask].clamp_min_(1e-8)
        denom = denom.abs().clamp_min(1e-8).sqrt()
        return ave / denom

    retr = expmap