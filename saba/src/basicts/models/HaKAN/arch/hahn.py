import torch
import torch.nn as nn


class HahnPolynomials(nn.Module):
    def __init__(self, input_dim, output_dim, degree, alpha, beta, N):
        super(HahnPolynomials, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.degree = degree
        self.a = alpha
        self.b = beta
        self.N = N

        self.hahn_coeffs = nn.Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.hahn_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

    def forward(self, x):
        four = False
        three = False
        two = False
        if len(x.shape) == 4:
            four = True
            a, b, c, d = x.shape
        elif len(x.shape) == 3:
            a, b, c = x.shape
            three = True
        else:
            a, b = x.shape
            two = True

        x = x.reshape(-1, self.input_dim)

        x = torch.tanh(x)

        hahn = torch.zeros(x.size(0), self.input_dim, self.degree + 1, device=x.device)

        hahn[:, :, 0] = 1

        if self.degree > 0:
            hahn[:, :, 1] = 1 - (((self.a + self.b + 2) * x )/((self.a + 1) * self.N))

        for n in range(2, self.degree + 1):
            m = n-1
            A = (m + self.a + self.b + 1) * (m + self.a + 1) * (self.N - m)
            A /= (m + m + self.a + self.b + 1)
            A /= (m + m + self.a + self.b + 2)

            C = m * (m + self.a + self.b + self.N + 1) * (m + self.b)
            C /= (m + m + self.a + self.b)
            C /= (m + m + self.a + self.b + 1)

            hahn[:, :, n] = ((A+C-x) * hahn[:, :, n - 1].clone()) - (C * hahn[:, :, n - 2].clone())
            hahn[:, :, n] /= A




        y = torch.einsum('bid,iod->bo', hahn, self.hahn_coeffs)

        if four == True:
            y = y.reshape(a, b, c, self.output_dim)
        elif three == True:
            y = y.reshape(a, b, self.output_dim)
        else:
            y = y.reshape(a, self.output_dim)

        return y