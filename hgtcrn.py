import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from loguru import logger


class ERB(nn.Module):
    def __init__(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        super().__init__()
        erb_filters = self.erb_filter_banks(
            erb_subband_1, erb_subband_2, nfft, high_lim, fs)
        nfreqs = nfft//2 + 1
        self.erb_subband_1 = erb_subband_1
        self.erb_fc = nn.Linear(nfreqs-erb_subband_1,
                                erb_subband_2, bias=False)
        self.ierb_fc = nn.Linear(
            erb_subband_2, nfreqs-erb_subband_1, bias=False)
        self.erb_fc.weight = nn.Parameter(erb_filters, requires_grad=False)
        self.ierb_fc.weight = nn.Parameter(erb_filters.T, requires_grad=False)

    def hz2erb(self, freq_hz):
        erb_f = 21.4*np.log10(0.00437*freq_hz + 1)
        return erb_f

    def erb2hz(self, erb_f):
        freq_hz = (10**(erb_f/21.4) - 1)/0.00437
        return freq_hz

    def erb_filter_banks(self, erb_subband_1, erb_subband_2, nfft=512, high_lim=8000, fs=16000):
        low_lim = erb_subband_1/nfft * fs
        erb_low = self.hz2erb(low_lim)
        erb_high = self.hz2erb(high_lim)
        erb_points = np.linspace(erb_low, erb_high, erb_subband_2)
        bins = np.round(self.erb2hz(erb_points)/fs*nfft).astype(np.int32)
        erb_filters = np.zeros(
            [erb_subband_2, nfft // 2 + 1], dtype=np.float32)

        erb_filters[0, bins[0]:bins[1]] = (bins[1] - np.arange(bins[0], bins[1]) + 1e-12) \
            / (bins[1] - bins[0] + 1e-12)
        for i in range(erb_subband_2-2):
            erb_filters[i + 1, bins[i]:bins[i+1]] = (np.arange(bins[i], bins[i+1]) - bins[i] + 1e-12)\
                / (bins[i+1] - bins[i] + 1e-12)
            erb_filters[i + 1, bins[i+1]:bins[i+2]] = (bins[i+2] - np.arange(bins[i+1], bins[i + 2]) + 1e-12) \
                / (bins[i + 2] - bins[i+1] + 1e-12)

        erb_filters[-1, bins[-2]:bins[-1]+1] = 1 - \
            erb_filters[-2, bins[-2]:bins[-1]+1]

        erb_filters = erb_filters[:, erb_subband_1:]
        return torch.from_numpy(np.abs(erb_filters))

    def bm(self, x):
        """x: (B,C,T,F)"""
        x_low = x[..., :self.erb_subband_1]
        x_high = self.erb_fc(x[..., self.erb_subband_1:])
        return torch.cat([x_low, x_high], dim=-1)

    def bs(self, x_erb):
        """x: (B,C,T,F_erb)"""
        x_erb_low = x_erb[..., :self.erb_subband_1]
        x_erb_high = self.ierb_fc(x_erb[..., self.erb_subband_1:])
        return torch.cat([x_erb_low, x_erb_high], dim=-1)


class SFE(nn.Module):
    """Subband Feature Extraction"""

    def __init__(self, kernel_size=3, stride=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(kernel_size=(1, kernel_size), stride=(
            1, stride), padding=(0, (kernel_size-1)//2))

    def forward(self, x):
        """x: (B,C,T,F)"""
        xs = self.unfold(x).reshape(
            x.shape[0], x.shape[1]*self.kernel_size, x.shape[2], x.shape[3])
        return xs


class TRA(nn.Module):
    """Temporal Recurrent Attention"""

    def __init__(self, channels):
        super().__init__()
        self.att_gru = nn.GRU(channels, channels*2, 1, batch_first=True)
        self.att_fc = nn.Linear(channels*2, channels)
        self.att_act = nn.Sigmoid()

    def forward(self, x):
        """x: (B,C,T,F)"""
        zt = torch.mean(x.pow(2), dim=-1)  # (B,C,T)
        at = self.att_gru(zt.transpose(1, 2))[0]
        at = self.att_fc(at).transpose(1, 2)
        at = self.att_act(at)
        At = at[..., None]  # (B,C,T,1)

        return x * At


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1, use_deconv=False, is_last=False):
        super().__init__()
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d
        self.conv = conv_module(in_channels, out_channels,
                                kernel_size, stride, padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.Tanh() if is_last else nn.PReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class GTConvBlock(nn.Module):
    """Group Temporal Convolution"""

    def __init__(self, in_channels, hidden_channels, kernel_size, stride, padding, dilation, use_deconv=False):
        super().__init__()
        self.use_deconv = use_deconv
        self.pad_size = (kernel_size[0]-1) * dilation[0]
        conv_module = nn.ConvTranspose2d if use_deconv else nn.Conv2d

        self.sfe = SFE(kernel_size=3, stride=1)

        self.point_conv1 = conv_module(in_channels//2*3, hidden_channels, 1)
        self.point_bn1 = nn.BatchNorm2d(hidden_channels)
        self.point_act = nn.PReLU()

        self.depth_conv = conv_module(hidden_channels, hidden_channels, kernel_size,
                                      stride=stride, padding=padding,
                                      dilation=dilation, groups=hidden_channels)
        self.depth_bn = nn.BatchNorm2d(hidden_channels)
        self.depth_act = nn.PReLU()

        self.point_conv2 = conv_module(hidden_channels, in_channels//2, 1)
        self.point_bn2 = nn.BatchNorm2d(in_channels//2)

        self.tra = TRA(in_channels//2)

    def shuffle(self, x1, x2):
        """x1, x2: (B,C,T,F)"""
        x = torch.stack([x1, x2], dim=1)
        x = x.transpose(1, 2).contiguous()  # (B,C,2,T,F)
        x = rearrange(x, 'b c g t f -> b (c g) t f')  # (B,2C,T,F)
        return x

    def forward(self, x):
        """x: (B, C, T, F)"""
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        x1 = self.sfe(x1)
        h1 = self.point_act(self.point_bn1(self.point_conv1(x1)))
        h1 = nn.functional.pad(h1, [0, 0, self.pad_size, 0])
        h1 = self.depth_act(self.depth_bn(self.depth_conv(h1)))
        h1 = self.point_bn2(self.point_conv2(h1))

        h1 = self.tra(h1)

        x = self.shuffle(h1, x2)

        return x


class GRNN(nn.Module):
    """Grouped RNN"""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn1 = nn.GRU(input_size//2, hidden_size//2, num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)
        self.rnn2 = nn.GRU(input_size//2, hidden_size//2, num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """
        if h == None:
            if self.bidirectional:
                h = torch.zeros(self.num_layers*2,
                                x.shape[0], self.hidden_size, device=x.device)
            else:
                h = torch.zeros(self.num_layers,
                                x.shape[0], self.hidden_size, device=x.device)
        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        h1, h2 = torch.chunk(h, chunks=2, dim=-1)
        h1, h2 = h1.contiguous(), h2.contiguous()
        y1, h1 = self.rnn1(x1, h1)
        y2, h2 = self.rnn2(x2, h2)
        y = torch.cat([y1, y2], dim=-1)
        h = torch.cat([h1, h2], dim=-1)
        return y, h


class DPGRNN(nn.Module):
    """Grouped Dual-path RNN"""

    def __init__(self, input_size, width, hidden_size, **kwargs):
        super(DPGRNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.width = width
        self.hidden_size = hidden_size

        self.intra_rnn = GRNN(input_size=input_size,
                              hidden_size=hidden_size//2, bidirectional=True)
        self.intra_fc = nn.Linear(hidden_size, hidden_size)
        self.intra_ln = nn.LayerNorm((width, hidden_size), eps=1e-8)

        self.inter_rnn = GRNN(input_size=input_size,
                              hidden_size=hidden_size, bidirectional=False)
        self.inter_fc = nn.Linear(hidden_size, hidden_size)
        self.inter_ln = nn.LayerNorm(((width, hidden_size)), eps=1e-8)

    def forward(self, x):
        """x: (B, C, T, F)"""
        # Intra RNN
        x = x.permute(0, 2, 3, 1)  # (B,T,F,C)
        intra_x = x.reshape(x.shape[0] * x.shape[1],
                            x.shape[2], x.shape[3])  # (B*T,F,C)
        intra_x = self.intra_rnn(intra_x)[0]  # (B*T,F,C)
        intra_x = self.intra_fc(intra_x)      # (B*T,F,C)
        intra_x = intra_x.reshape(
            x.shape[0], -1, self.width, self.hidden_size)  # (B,T,F,C)
        intra_x = self.intra_ln(intra_x)
        intra_out = torch.add(x, intra_x)

        # Inter RNN
        x = intra_out.permute(0, 2, 1, 3)  # (B,F,T,C)
        inter_x = x.reshape(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        inter_x = self.inter_rnn(inter_x)[0]  # (B*F,T,C)
        inter_x = self.inter_fc(inter_x)      # (B*F,T,C)
        inter_x = inter_x.reshape(
            x.shape[0], self.width, -1, self.hidden_size)  # (B,F,T,C)
        inter_x = inter_x.permute(0, 2, 1, 3)   # (B,T,F,C)
        inter_x = self.inter_ln(inter_x)
        inter_out = torch.add(intra_out, inter_x)

        dual_out = inter_out.permute(0, 3, 1, 2)  # (B,C,T,F)

        return dual_out


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.en_convs = nn.ModuleList([
            # groups=1, use_deconv=False, is_last=False
            ConvBlock(3*3, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=1, use_deconv=False, is_last=False),
            ConvBlock( 16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=False, is_last=False),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1), use_deconv=False),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1), use_deconv=False),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1), use_deconv=False)
        ])

    def forward(self, x):
        en_outs = []
        for i in range(len(self.en_convs)):
            logger.info(f'Encoder {i} {x.shape}')
            x = self.en_convs[i](x)
            logger.info(f'Encoder {i} {x.shape}')
            en_outs.append(x)
        return x, en_outs


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.de_convs = nn.ModuleList([
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(2*5, 1), dilation=(5, 1), use_deconv=True),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(2*2, 1), dilation=(2, 1), use_deconv=True),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(2*1, 1), dilation=(1, 1), use_deconv=True),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2, use_deconv=True, is_last=False),
            ConvBlock(16,  2, (1, 5), stride=(1, 2), padding=(0, 2), groups=1, use_deconv=True, is_last=True)
        ])

    def forward(self, x, en_outs):
        N_layers = len(self.de_convs)
        for i in range(N_layers):
            logger.info(f'Decoder {i} {x.shape}')
            x = self.de_convs[i](x + en_outs[N_layers-1-i])
            logger.info(f'Decoder {i} {x.shape}')
        return x


class Mask(nn.Module):
    """Complex Ratio Mask"""

    def __init__(self):
        super().__init__()

    def forward(self, mask, spec):
        s_real = spec[:, 0] * mask[:, 0] - spec[:, 1] * mask[:, 1]
        s_imag = spec[:, 1] * mask[:, 0] + spec[:, 0] * mask[:, 1]
        s = torch.stack([s_real, s_imag], dim=1)  # (B,2,T,F)
        return s


class AuxIVA(nn.Module):
    """
    简化版 AuxIVA，仅作结构占位，实际可替换为更复杂实现。
    输入: (B, C, F, T, 2) 复数谱
    输出: (B, C, F, T, 2) 分离信号
    """
    def __init__(self):
        super().__init__()

    def forward(self, spec_in):
        # 简单平均模拟分离
        spec_real = spec_in[..., 0]  # (B, C, F, T)
        spec_imag = spec_in[..., 1]  # (B, C, F, T)
        separated_real = torch.mean(spec_real, dim=1, keepdim=True)  # (B, 1, F, T)
        separated_imag = torch.mean(spec_imag, dim=1, keepdim=True)  # (B, 1, F, T)
        separated = torch.cat([separated_real, separated_imag], dim=1)  # (B, 2, F, T)
        # 拼成 (B, 2, F, T, 2)
        separated = torch.stack([separated_real, separated_imag], dim=-1).repeat(1,2,1,1,1)
        return separated

class FeatureSelect(nn.Module):
    """
    FS模块，输入(B, C, F, T, 2)，输出(B, C*3, T, F)
    """
    def __init__(self):
        super().__init__()
    def forward(self, spec):
        # (B, C, F, T, 2) -> (B, C, T, F, 2)
        spec = spec.permute(0,1,3,2,4)
        spec_real = spec[..., 0]
        spec_imag = spec[..., 1]
        spec_mag = torch.sqrt(spec_real**2 + spec_imag**2 + 1e-12)
        feat = torch.stack([spec_mag, spec_real, spec_imag], dim=2)  # (B, C, 3, T, F)
        feat = feat.view(spec.shape[0], spec.shape[1]*3, spec.shape[2], spec.shape[3])  # (B, C*3, T, F)
        logger.info(f'FeatureSelect输出 shape: {feat.shape}')
        return feat

class SingleEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce_conv = nn.Conv2d(36, 9, kernel_size=1)  # 18+18=36
        self.encoder = Encoder()
    def forward(self, x):
        logger.info(f'SingleEncoder concat后 shape: {x.shape}')
        x = self.reduce_conv(x)
        logger.info(f'SingleEncoder 1x1 conv后 shape: {x.shape}')
        f, outs = self.encoder(x)
        return f, outs

class DualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce_conv1 = nn.Conv2d(18, 9, kernel_size=1)
        self.reduce_conv2 = nn.Conv2d(18, 9, kernel_size=1)
        self.encoder1 = Encoder()
        self.encoder2 = Encoder()
        self.f_conv = nn.Conv2d(32, 16, kernel_size=1)
        self.out_conv = nn.Conv2d(32, 16, kernel_size=1)
    def forward(self, x1, x2):
        logger.info(f'DualEncoder x1 shape: {x1.shape}, x2 shape: {x2.shape}')
        x1 = self.reduce_conv1(x1)
        x2 = self.reduce_conv2(x2)
        logger.info(f'DualEncoder x1 reduce后 shape: {x1.shape}, x2 reduce后 shape: {x2.shape}')
        f1, outs1 = self.encoder1(x1)
        f2, outs2 = self.encoder2(x2)
        f = torch.cat([f1, f2], dim=1)
        f = self.f_conv(f)
        outs = [self.out_conv(torch.cat([o1, o2], dim=1)) for o1, o2 in zip(outs1, outs2)]
        return f, outs

class HGTCRN(nn.Module):
    def __init__(self, num_freqs, dual_encoder=True, masking_mode="iva"):
        super().__init__()
        self.dual_encoder = dual_encoder
        self.masking_mode = masking_mode  # "iva" or "noisy"
        self.erb = ERB(65, 64, nfft=(num_freqs-1)*2)
        self.sfe = SFE(3, 1)
        self.fs = FeatureSelect()
        if dual_encoder:
            self.encoder = DualEncoder()
        else:
            self.encoder = SingleEncoder()
        self.dpgrnn1 = DPGRNN(16, 33, 16)
        self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = Decoder()
        self.mask = Mask()
        self.iva = AuxIVA()

    def forward(self, spec_in):
        B, C, F, T, _ = spec_in.shape
        Y = spec_in
        Yiva = self.iva(spec_in)
        logger.info(f'Y shape: {Y.shape}, Yiva shape: {Yiva.shape}')
        Y_feat = self.sfe(self.erb.bm(self.fs(Y)))   # (B, 6, T, F')
        Yiva_feat = self.sfe(self.erb.bm(self.fs(Yiva)))
        logger.info(f'Y_feat shape: {Y_feat.shape}, Yiva_feat shape: {Yiva_feat.shape}')
        if self.dual_encoder:
            feat, en_outs = self.encoder(Y_feat, Yiva_feat)
        else:
            feat, en_outs = self.encoder(torch.cat([Y_feat, Yiva_feat], dim=1))
        logger.info(f'Encoder输出 shape: {feat.shape}')
        feat = self.dpgrnn1(feat)
        logger.info(f'dpgrnn1输出 shape: {feat.shape}')
        feat = self.dpgrnn2(feat)
        logger.info(f'dpgrnn2输出 shape: {feat.shape}')
        m_feat = self.decoder(feat, en_outs)
        m = self.erb.bs(m_feat)
        logger.info(f'mask特征 shape: {m.shape}')
        if self.masking_mode == "iva":
            ref_spec = Yiva
            logger.info("Masking on IVA output.")
        else:
            ref_spec = Y
            logger.info("Masking on noisy input.")
        # ref_spec: [B, C, F, T, 2]
        ref_spec = ref_spec[:, :2]                  # [B, 2, F, T, 2]
        ref_spec = ref_spec.permute(0, 1, 3, 2, 4)  # [B, 2, T, F, 2]
        ref_spec = ref_spec[..., 0]                 # [B, 2, T, F]
        logger.info(f'ref_spec shape for mask: {ref_spec.shape}')
        spec_enh = self.mask(m, ref_spec)
        spec_enh = spec_enh.permute(0,3,2,1)
        logger.info(f'最终输出 shape: {spec_enh.shape}')
        return spec_enh

if __name__ == "__main__":
    # 测试输入
    in_ch2_wav = torch.randn(2, 16000) # (C=2, T=16000)
    in_ch2_spec = torch.stft(
        in_ch2_wav, 
        n_fft=512, 
        hop_length=256, 
        win_length=512, 
        window=torch.hann_window(512).pow(0.5),
        return_complex=True)
    # logger.info(in_ch2_spec.shape)     # (C=2, F=257, T=63)

    in_ch2_spec = torch.view_as_real(in_ch2_spec) # (C=2, F=257, T=63, 2)
    # logger.info(in_ch2_spec.shape)

    in_ch2_spec = in_ch2_spec.unsqueeze(0) # B=1 C=2 F=257 T=63 2
    logger.info(in_ch2_spec.shape)

    # in_ch2_spec = torch.randn(1, 2, 257, 63, 2)

    model = HGTCRN(num_freqs=in_ch2_spec.shape[2], dual_encoder=False, masking_mode="iva")
    out = model(in_ch2_spec)
    print("输入 shape:", in_ch2_spec.shape)      # (1, 2, 257, 63, 2)
    print("输出 shape:", out.shape)             # (1, 257, 63, 2)

    out_real = out[..., 0]
    out_imag = out[..., 1]
    out_spec = torch.complex(out_real, out_imag)

    out_ch2_wav = torch.istft(
        out_spec,
        n_fft=512,
        hop_length=256,
        win_length=512,
        window=torch.hann_window(512).pow(0.5),
    )
    logger.info(out_ch2_wav.shape)

    """complexity count"""
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model, (2, 257, 63, 2), as_strings=True,
                                              print_per_layer_stat=False, verbose=True)
    print(flops, params)

    # model = HGTCRN(num_freqs=in_ch2_spec.shape[2], dual_encoder=True, masking_mode="iva")
    # model = HGTCRN(num_freqs=in_ch2_spec.shape[2], dual_encoder=True, masking_mode="noisy")
    # 61.52 MMac 33.01 k

    # model = HGTCRN(num_freqs=in_ch2_spec.shape[2], dual_encoder=False, masking_mode="noisy")
    # model = HGTCRN(num_freqs=in_ch2_spec.shape[2], dual_encoder=False, masking_mode="iva")
    # 43.5 MMac 24.0 k
