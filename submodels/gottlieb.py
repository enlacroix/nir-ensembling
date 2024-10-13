import torch
import torch.nn as nn


def gottlieb(n, x, alpha):
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 2 * alpha * x
    else:
        return 2 * (alpha + n - 1) * x * gottlieb(n - 1, x, alpha) - (alpha + 2 * n - 2) * gottlieb(n - 2, x, alpha)


class GottliebKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree, use_layernorm):
        super(GottliebKANLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.use_layernorm = use_layernorm
        self.layernorm = nn.LayerNorm(output_dim)
        self.alpha = nn.Parameter(torch.randn(1))
        self.gottlieb_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.gottlieb_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        x = torch.sigmoid(x)

        gottlieb_basis = []
        for n in range(self.degree + 1):
            gottlieb_basis.append(gottlieb(n, x, self.alpha))
        gottlieb_basis = torch.stack(gottlieb_basis, dim=-1)

        y = torch.einsum("bid,iod->bo", gottlieb_basis, self.gottlieb_coeffs)
        y = y.view(-1, self.output_dim)

        if self.use_layernorm:
            y = self.layernorm(y)

        return y


class GottliebKAN(nn.Module):
    def __init__(
            self,
            layers_hidden,
            spline_order=3,
    ):
        super(GottliebKAN, self).__init__()
        self.layers_hidden = layers_hidden
        self.spline_order = spline_order
        self.layers = torch.nn.ModuleList()
        layers_hidden = [layers_hidden[0]] + sum([[x] * 2 for x in layers_hidden[1:-1]], []) + [layers_hidden[-1]]

        for input_dim, output_dim in zip(layers_hidden, layers_hidden[1:-1]):
            self.layers.append(
                GottliebKANLayer(
                    input_dim,
                    output_dim,
                    degree=spline_order,
                    use_layernorm=True
                )
            )

        self.layers.append(
            GottliebKANLayer(
                layers_hidden[-2],
                layers_hidden[-1],
                degree=spline_order,
                use_layernorm=False
            )
        )

    def forward(self, x: torch.Tensor):
        x = x.view(-1, self.layers_hidden[0])
        for layer in self.layers:
            x = layer(x)
        return x
