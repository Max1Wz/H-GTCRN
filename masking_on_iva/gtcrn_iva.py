import torch
import numpy as np
import torch.nn as nn
from einops import rearrange


def multi_channel_stft(x, n_fft, hop_length, win_length, window, onesided=True):
    """
    x: (B, L, C)
    X: (B, C, F, T)
    """
    bs = x.shape[0]
    x = x.transpose(1, 2).reshape(-1, x.size(1))

    X = torch.stft(x, n_fft, hop_length, win_length, window, onesided=onesided, return_complex=True)
    X = torch.view_as_real(X)
    X = X.view(bs, -1, X.shape[1], X.shape[2], 2)
    X = torch.complex(X[..., 0], X[..., 1])

    return X


def fd_wpe(X, rt60, shift, D=2, fs=16000, num_iter=10):
    """Frequency domain weighted prediction error (FD-WPE)"""
    B, M, F, T = X.shape
    device = X.device
    eps = 1e-3 * torch.mean(torch.max(torch.max(X.abs() ** 2, dim=-1).values, dim=-2).values).to(device)

    Lg = int(rt60 * fs / shift)
    Xp = torch.permute(X, [0, 2, 1, 3])
    eyes = torch.tile(torch.eye(M * Lg, M * Lg, dtype=X.dtype), (B, F, 1, 1)).to(device)

    X_delay = torch.zeros(B, F, M * Lg, T, dtype=X.dtype).to(device)
    for l in range(Lg):
        X_delay[:, :, l * M : (l + 1) * M, D + l : T] = Xp[:, :, :, 0 : T - D - l]

    Y = Xp.clone().to(device)

    for _ in range(num_iter):
        lambdaa = torch.max(torch.mean(torch.abs(Y) ** 2, dim=-2, keepdim=True), eps).to(device)

        temp = X_delay / lambdaa

        R = temp @ torch.conj(X_delay.transpose(-2, -1))
        P = temp @ torch.conj(Xp.transpose(-2, -1))

        G = torch.linalg.inv(R + eps * eyes) @ P

        Y = Xp - torch.conj(G.transpose(-2, -1)) @ X_delay

    return Y.permute(0, 2, 1, 3)


def projection_back(Y, ref):
    device = Y.device
    num = torch.sum(torch.conj(ref[:, :, :, None]) * Y, dim=1)
    denom = torch.sum(torch.abs(Y) ** 2, dim=1)

    c = torch.ones(num.shape, dtype=Y.dtype).to(device)
    I = denom > 0.0
    c[I] = num[I] / denom[I]
    
    return c


def auxiva(X, n_src=None, n_iter=20, proj_back=True, W0=None, model="laplace"):
    """Auxiliary-function-based Independent Vector Analysis (AuxIVA)"""
    n_batches, n_frames, n_freq, n_chan = X.shape
    device = X.device
    if n_src is None:
        n_src = n_chan

    W_hat = torch.zeros((n_batches, n_freq, n_chan, n_chan), dtype=X.dtype).to(device)
    W = W_hat[:, :, :n_src, :]

    if W0 is None:
        W[:, :, :, :n_src] = torch.tile(
            torch.eye(n_src, n_src, dtype=X.dtype), (n_batches, n_freq, 1, 1)
        ).to(device)
    else:
        W[:, :, :, :] = W0

    eps = 1e-10
    eyes = torch.tile(
        torch.eye(n_chan, n_chan, dtype=X.dtype), (n_batches, n_freq, 1, 1)
    ).to(device)

    r_inv = torch.zeros((n_batches, n_src, n_frames)).to(device)
    r = torch.zeros((n_batches, n_src, n_frames)).to(device)

    Y = torch.zeros((n_batches, n_freq, n_src, n_frames), dtype=X.dtype).to(device)

    X_original = X
    X = X.permute(0, 2, 3, 1).clone()

    def demix(Y, X, W):
        Y[:, :, :, :] = torch.matmul(W, X)

    for _ in range(n_iter):
        demix(Y, X, W)

        if model == "laplace":
            r[:, :, :] = 2.0 * torch.norm(Y, dim=1)
        elif model == "gauss":
            r[:, :, :] = (torch.norm(Y, dim=1) ** 2) / n_freq

        r[r < eps] = eps
        r_inv[:, :, :] = 1.0 / r

        for s in range(n_src):
            V = (
                torch.matmul(
                    (X * r_inv[:, None, s, None, :]), torch.conj(X.swapaxes(2, 3))
                )
                / n_frames
            )

            WV = torch.matmul(W_hat, V)
            W[:, :, s, :] = torch.conj(
                torch.linalg.solve(WV + eps * eyes, eyes[:, :, :, s])
            )

            denom = torch.matmul(
                torch.matmul(W[:, :, None, s, :], V[:, :, :, :]),
                torch.conj(W[:, :, s, :, None]),
            )
            W[:, :, s, :] /= torch.sqrt(
                denom[:, :, :, 0] + eps * torch.ones((n_batches, n_freq, 1)).to(device)
            )

    demix(Y, X, W)

    Y = Y.permute(0, 3, 1, 2).clone()

    if proj_back:
        z = projection_back(Y, X_original[:, :, :, 0])
        Y *= torch.conj(z[:, None, :, :])
    
    return Y


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft // 2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs - erb_subband_1, erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(erb_subband_2, nfreqs - erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 24.7 * np.log10(0.00437 * freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10 ** (erb_f / 24.7) - 1) / 0.00437
        return freq_hz

    def erb_filter_banks(
        self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000
    ):
        low_lim = erb_subband_1 / nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points) / fs * nfft).astype(np.int32)
        erb_filters = np.zeros([erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0] : bins[1]] = (
            bins[1] - np.arange(bins[0], bins[1]) + 1e-12
        ) / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2 - 2):
            erb_filters[i + 1, bins[i] : bins[i + 1]] = (
                np.arange(bins[i], bins[i + 1]) - bins[i] + 1e-12
            ) / (bins[i + 1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i + 1] : bins[i + 2]] = (
                bins[i + 2] - np.arange(bins[i + 1], bins[i + 2]) + 1e-12
            ) / (bins[i + 2] - bins[i + 1] + 1e-12)

        erb_filters[-1, bins[-2] : bins[-1] + 1] = (
            1 - erb_filters[-2, bins[-2] : bins[-1] + 1]
        )

        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """x: (B, C, T, F)"""
        x_low = x[..., : self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1 :])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """x: (B, C, T, F_erb)"""
        x_erb_low = x_erb[..., : self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1 :])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""
    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, (kernel_size - 1) // 2),
        )

    def forward(self, x):
        """x: (B, C, T, F)"""
        xs = self.unfold(x).reshape(
            x.shape[0], x.shape[1] * self.kernel_size, x.shape[2], x.shape[3]
        )
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""
    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels * 2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels * 2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        zt = torch.mean(x.pow(2), dim=-1)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        At = self.att_act(at)[..., None]
        
        return x * At


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        use_deconv=False,
    ):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation=dilation,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""
    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        self.pad_size = (kernel_size[0] - 1) * dilation[0]
        self.sfe = SFE(kernel_size=3, stride=1)
        self.point_conv1 = ConvBlock(in_channels // 2 * 3, hidden_channels, 1)
        self.depth_conv = ConvBlock(
            hidden_channels,
            hidden_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=hidden_channels,
        )
        self.point_conv2 = ConvBlock(hidden_channels, in_channels // 2, 1)
        self.point_conv2.act = nn.Identity()
        self.tra = TRA(in_channels // 2)

    def shuffle(self, x1, x2):
        """x1, x2: (B, C, T, F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()
        x = rearrange(x, "b c g t f -> b (c g) t f")
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_conv1(x1)
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_conv(h1)
        h1 = self.point_conv2(h1)
        h1 = self.tra(h1)

        x = self.shuffle(h1, x2)

        return x


class GRNN(nn.Module):
    """Grouped RNN"""
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )
        self.rnn2 = nn.GRU(
            input_size // 2,
            hidden_size // 2,
            num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h is None:
            if self.bidirectional:
                h = torch.zeros(
                    self.num_layers * 2, x.shape[0], self.hidden_size, device=x.device
                )
            else:
                h = torch.zeros(
                    self.num_layers, x.shape[0], self.hidden_size, device=x.device
                )
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        y1, h1 = self.rnn1(x1, h1.contiguous())
        y2, h2 = self.rnn2(x2, h2.contiguous())
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        
        return y, h


class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""
    def __init__(self, input_size, width, hidden_size, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(
            input_size=input_size, hidden_size=hidden_size // 2, bidirectional=True
        )
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(
            input_size=input_size, hidden_size=hidden_size, bidirectional=False
        )
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F)"""

        x = x.permute(0, 2, 3, 1)
        intra_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        intra_x = self.intra_rnn(intra_x)[0]
        intra_x = self.intra_fc(intra_x)
        intra_x = intra_x.reshape(x.shape[0], -1, self.width, self.hidden_size)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        x = intra_out.permute(0, 2, 1, 3)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_rnn(inter_x)[0]
        inter_x = self.inter_fc(inter_x)
        inter_x = inter_x.reshape(x.shape[0], self.width, -1, self.hidden_size)
        inter_x = inter_x.permute(0, 2, 1, 3)
        inter_x = self.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)

        dual_out = inter_out.permute(0, 3, 1, 2)

        return dual_out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList(
            [
                ConvBlock(6 * 3, 16, (1, 5), stride=(1, 2), padding=(0, 2)),
                ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
                GTConvBlock(
                    16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)
                ),
                GTConvBlock(
                    16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)
                ),
                GTConvBlock(
                    16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)
                ),
            ]
        )

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            x = self.en_convs[i](x)
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList(
            [
                GTConvBlock(
                    16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)
                ),
                GTConvBlock(
                    16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)
                ),
                GTConvBlock(
                    16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)
                ),
                ConvBlock(
                    16,
                    16,
                    (1, 5),
                    stride=(1, 2),
                    padding=(0, 2),
                    groups=2,
                    use_deconv=True,
                ),
                ConvBlock(
                    16, 2, (1, 5), stride=(1, 2), padding=(0, 2), use_deconv=True
                ),
            ]
        )
        self.de_convs[-1].act = nn.Tanh()

    def forward(self, x, en_outs):
        N = len(self.de_convs)
        for i in range(N):
            x = self.de_convs[i](x + en_outs[N - 1 - i])
        return x


class Mask(nn.Module):
    """Complex Ratio Mask"""
    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        s = torch.stack([s_real, s_imag], dim=1)
        
        return s


class GTCRN_IVA(nn.Module):
    """Masking-on-IVA: CRM is applied to the energy-selected AuxIVA speech channel.

    DSP matches gtcrn_iva1.batch_auxiva: FD-WPE then AuxIVA with n_iter=20.
    Inference returns a single enhanced waveform (the trained enhanced channel).
    """

    def __init__(self):
        super().__init__()
        self.n_fft = 512
        self.hop_len = 256
        self.win_len = 512

        self.erb = ERB(65, 64)
        self.sfe = SFE(3, 1)
        self.encoder = Encoder()
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = Decoder()
        self.mask = Mask()

    def forward(self, x):
        """x: (B, C, L)"""
        device = x.device
        n_samples = x.shape[-1]

        stft_kwargs = {
            "n_fft": self.n_fft,
            "hop_length": self.hop_len,
            "win_length": self.win_len,
            "window": torch.hann_window(self.win_len).to(device),
            "onesided": True,
        }
        spec_orig = multi_channel_stft(x.transpose(1, 2), **stft_kwargs)

        spec_drb = fd_wpe(spec_orig, rt60=0.3, shift=256, D=2, fs=16000, num_iter=1)
        spec_2ch = auxiva(spec_drb.transpose(1, 3), n_iter=20).transpose(1, 3)

        spec_norm = torch.norm(spec_2ch, dim=(2, 3))
        pred = torch.where(spec_norm[:, 0] < spec_norm[:, 1], 1, 0)
        pred = pred.view(-1, 1, 1, 1)
        spec_selected = spec_2ch[:, 0] * pred[:, 0] + spec_2ch[:, 1] * (1 - pred[:, 0])
        spec_unselected = spec_2ch[:, 1] * pred[:, 0] + spec_2ch[:, 0] * (1 - pred[:, 0])

        spec_sel = torch.view_as_real(spec_selected).permute(0, 3, 2, 1)
        spec_sel_mag = torch.norm(spec_sel, dim=1, keepdim=True).clamp(1e-12)
        spec_sel_log = torch.log10(spec_sel_mag)

        spec_un = torch.view_as_real(spec_unselected).permute(0, 3, 2, 1)
        spec_un_mag = torch.norm(spec_un, dim=1, keepdim=True).clamp(1e-12)
        spec_un_log = torch.log10(spec_un_mag)

        spec = torch.view_as_real(spec_orig)
        spec = rearrange(spec, "b c f t ri -> b (c ri) t f")

        feat = torch.cat([spec, spec_sel_log, spec_un_log], dim=1)

        feat = self.erb.bm(feat)
        feat = self.sfe(feat)
        feat, en_outs = self.encoder(feat)
        feat = self.dpgrnn1(feat)
        feat = self.dpgrnn2(feat)
        m_feat = self.decoder(feat, en_outs)
        m = self.erb.bs(m_feat)

        spec_iva = torch.view_as_real(spec_selected[:, None])
        spec_iva = rearrange(spec_iva, "b c f t ri -> b (c ri) t f")
        spec_enh = self.mask(m, spec_iva)

        spec_enh = spec_enh.permute(0, 3, 2, 1)
        spec_enh = torch.complex(spec_enh[..., 0], spec_enh[..., 1])

        output = torch.istft(spec_enh, **stft_kwargs)
        output = torch.nn.functional.pad(output, (0, n_samples - output.shape[-1]))

        return output


if __name__ == "__main__":
    model = GTCRN_IVA().eval()

    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(
        model,
        (2, 16000),
        as_strings=True,
        print_per_layer_stat=False,
        verbose=True,
    )
    print(f"Computational complexity: {flops}, Parameters: {params}")

    x = torch.randn(3, 2, 16000)
    y = model(x)
    print(y.shape)