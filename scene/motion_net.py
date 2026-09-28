import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder


class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, leakyReLU=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        if leakyReLU:
            self.act = nn.LeakyReLU(0.02)
        else:
            self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=64, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len, out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attentionConvNet(y)
        y = self.attentionNet(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1)  # [1, dim_aud]


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(dim_in, 32 if dim_in < 128 else 128, kernel_size=3, stride=2, padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32 if dim_in < 128 else 128, 32 if dim_in < 128 else 128, kernel_size=3, stride=2, padding=1,
                      bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32 if dim_in < 128 else 128, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size / 2)
        x = x[:, :, 8 - half_w:8 + half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x)
        return x


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

    def forward(self, x):
        out = self.audio_encoder(x)
        out = out.squeeze(2).squeeze(2)

        return out


# Audio feature extractor
class AudioNet_ave(nn.Module):
    def __init__(self, dim_in=29, dim_aud=64, win_size=16):
        super(AudioNet_ave, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, dim_aud),
        )

    def forward(self, x):
        # half_w = int(self.win_size/2)
        # x = x[:, :, 8-half_w:8+half_w]
        # x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x).permute(1, 0, 2).squeeze(0)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden,
                                 self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=False))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
                # x = F.dropout(x, p=0.1, training=self.training)

        return x


class MotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim=32,
                 ind_dim=0,
                 args=None,
                 ):
        super(MotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        elif 'ave' in args.audio_extractor:
            self.audio_in_dim = 32
        else:
            raise NotImplementedError

        self.bound = 0.15
        self.exp_eye = True

        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1)

            # audio network
        self.audio_dim = audio_dim
        if args.audio_extractor == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=16,
                                                      log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=16,
                                                      log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=16,
                                                      log2_hashmap_size=17, desired_resolution=256 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        self.num_layers = 3
        self.hidden_dim = 64

        self.exp_in_dim = 6 - 1
        self.eye_dim = 6 if self.exp_eye else 0
        self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)

        self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)

        # rot: 4   xyz: 3   opac: 1  scale: 3
        self.out_dim = 11
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim,
                             self.hidden_dim, self.num_layers)

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)

        self.cache = None

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:, :1], x[:, -1:]], dim=-1)
        return xy, yz, xz

    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)

        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a)  # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0))  # [1, 64]

        return enc_a

    def forward(self, x, a, e=None, c=None):
        # x: [N, 3], in [-bound, bound]
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att

        eye_att = torch.relu(self.eye_att_net(enc_x))
        enc_e = self.exp_encode_net(e[:-1])
        enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
        enc_e = enc_e * eye_att
        if c is not None:
            c = c.repeat(enc_x.shape[0], 1)
            h = torch.cat([enc_x, enc_w, enc_e, c], dim=-1)
        else:
            h = torch.cat([enc_x, enc_w, enc_e], dim=-1)

        h = self.sigma_net(h)

        d_xyz = h[..., :3] * 1e-2
        d_rot = h[..., 3:7]
        d_opa = h[..., 7:8]
        d_scale = h[..., 8:11]

        results = {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud': aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye': eye_att.norm(dim=-1, keepdim=True),
        }
        self.cache = results
        return results

    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})

        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.eye_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})
        params.append({'params': self.exp_encode_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params


class MouthMotionNetwork(nn.Module):
    def __init__(self,
                 audio_dim=32,
                 ind_dim=0,
                 args=None,
                 ):
        super(MouthMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        elif 'ave' in args.audio_extractor:
            self.audio_in_dim = 32
        else:
            raise NotImplementedError

        self.bound = 0.15

        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1)

            # audio network
        self.audio_dim = audio_dim
        if args.audio_extractor == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)

        # DYNAMIC PART
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=64,
                                                      log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=64,
                                                      log2_hashmap_size=17, desired_resolution=384 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=64,
                                                      log2_hashmap_size=17, desired_resolution=384 * self.bound)

        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        ## sigma network
        self.num_layers = 3
        self.hidden_dim = 32

        self.out_dim = 7
        self.move_dim = 3
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.individual_dim + self.move_dim, self.out_dim,
                             self.hidden_dim, self.num_layers)
        self.scaler_net = MLP(self.in_dim + self.move_dim, 1, 16, 3)

        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)

    def encode_audio(self, a):
        # a: [1, 29, 16] or [8, 29, 16], audio features from deepspeech
        # if emb, a should be: [1, 16] or [8, 16]

        # fix audio traininig
        if a is None: return None

        enc_a = self.audio_net(a)  # [1/8, 64]
        enc_a = self.audio_att_net(enc_a.unsqueeze(0))  # [1, 64]

        return enc_a

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:, :1], x[:, -1:]], dim=-1)
        return xy, yz, xz

    def encode_x(self, xyz, bound):
        # x: [N, 3], in [-bound, bound]
        N, M = xyz.shape
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)

        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)

    def forward(self, x, a, move):
        # x: [N, 3], in [-bound, bound]
        # move: [1, D]
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_w = enc_a.repeat(enc_x.shape[0], 1)
        # move = torch.as_tensor([[move_max, move_min]]).repeat(enc_x.shape[0], 1).cuda()
        move = move.repeat(enc_x.shape[0], 1)

        h = torch.cat([enc_x, enc_w, move], dim=-1)
        h = self.sigma_net(h)

        h_s = torch.cat([enc_x, move], dim=-1)
        h_s = self.scaler_net(h_s)

        d_xyz = h[..., :3] * 1e-2
        d_xyz[..., 0] = d_xyz[..., 0] / 5
        d_xyz[..., 2] = d_xyz[..., 2] / 5
        d_rot = h[..., 3:]
        return {
            'd_xyz': d_xyz * torch.sigmoid(h_s) * 2,
            'd_rot': d_rot,
            # 'ambient_aud' : aud_ch_att.norm(dim=-1, keepdim=True),
        }

    # optimizer utils
    def get_params(self, lr, lr_net, wd=0):

        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.scaler_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        params.append({'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})

        params.append({'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': wd})

        return params

class PersonalizedMotionNetwork(nn.Module):
    def __init__(self, audio_dim=32, ind_dim=0, args=None):
        super(PersonalizedMotionNetwork, self).__init__()

        if 'esperanto' in args.audio_extractor:
            self.audio_in_dim = 44
        elif 'deepspeech' in args.audio_extractor:
            self.audio_in_dim = 29
        elif 'hubert' in args.audio_extractor:
            self.audio_in_dim = 1024
        elif 'ave' in args.audio_extractor:
            self.audio_in_dim = 32
        else:
            raise NotImplementedError

        self.args = args
        self.bound = 0.15
        self.exp_eye = args.type == 'face'

        self.individual_dim = ind_dim
        if self.individual_dim > 0:
            self.individual_codes = nn.Parameter(torch.randn(10000, self.individual_dim) * 0.1)

        # Audio network
        self.audio_dim = audio_dim
        if args.audio_extractor == 'ave':
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, self.audio_dim)
        self.audio_att_net = AudioAttNet(self.audio_dim)

        # Dynamic spatial encoding
        self.num_levels = 12
        self.level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=16,
                                                      log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_yz, self.in_dim_yz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=16,
                                                      log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.encoder_xz, self.in_dim_xz = get_encoder('hashgrid', input_dim=2, num_levels=self.num_levels,
                                                      level_dim=self.level_dim, base_resolution=16,
                                                      log2_hashmap_size=17, desired_resolution=256 * self.bound)
        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz

        self.num_layers = 3
        self.hidden_dim = 32 if args.type == 'face' else 16

        self.exp_in_dim = 5
        self.eye_dim = 6 if self.exp_eye else 0
        if self.exp_eye:
            self.exp_encode_net = MLP(self.exp_in_dim, self.eye_dim - 1, 16, 2)
            self.eye_att_net = MLP(self.in_dim, self.eye_dim, 16, 2)

        # rot: 4, xyz: 3, opacity: 1, scale: 3
        self.out_dim = 11 if args.type == 'face' else 7
        self.sigma_net = MLP(self.in_dim + self.audio_dim + self.eye_dim + self.individual_dim, self.out_dim,
                             self.hidden_dim, self.num_layers)
        self.align_net = MLP(self.in_dim, 6, self.hidden_dim, 2)
        self.aud_ch_att_net = MLP(self.in_dim, self.audio_dim, 32, 2)

        # ============================================================
        # PPR: Phoneme-Conditioned Personalized Residual
        # Kept for ablation. Do not enable together with CA-MLVSR.
        # ============================================================
        self.use_phoneme_residual = getattr(args, 'use_phoneme_residual', False) and args.type == 'face'
        self.num_phonemes = getattr(args, 'num_phonemes', 64)
        self.phoneme_rank = getattr(args, 'phoneme_rank', 8)
        self.phoneme_res_scale = getattr(args, 'phoneme_res_scale', 1.0)

        if self.use_phoneme_residual:
            self.phoneme_coeff = nn.Embedding(self.num_phonemes, self.phoneme_rank, padding_idx=0)
            nn.init.zeros_(self.phoneme_coeff.weight)
            self.phoneme_basis_net = MLP(self.in_dim, self.phoneme_rank * 3, self.hidden_dim, 2)

        # ============================================================
        # Shared Viseme Residual + Mouth Localization
        # ============================================================
        self.use_viseme_shared_residual = getattr(args, 'use_viseme_shared_residual', False) and args.type == 'face'
        self.num_visemes = getattr(args, 'num_visemes', 9)
        self.viseme_rank = getattr(args, 'viseme_rank', 8)
        self.viseme_res_scale = getattr(args, 'viseme_res_scale', 1.0)
        self.viseme_gate_gamma = getattr(args, 'viseme_gate_gamma', 2.0)

        # ============================================================
        # Coverage-Adaptive Completion
        # alpha = 1 - C_eff * n_p / (n_p + tau)
        # ============================================================
        self.use_coverage_adaptive = getattr(args, 'use_coverage_adaptive', False) and self.use_viseme_shared_residual
        self.coverage_tau = getattr(args, 'coverage_tau', 4.0)

        if self.use_viseme_shared_residual:
            self.viseme_coeff = nn.Embedding(self.num_visemes, self.viseme_rank, padding_idx=0)
            nn.init.zeros_(self.viseme_coeff.weight)
            self.viseme_basis_net = MLP(self.in_dim, self.viseme_rank * 3, self.hidden_dim, 2)

    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        xy, yz, xz = x[:, :-1], x[:, 1:], torch.cat([x[:, :1], x[:, -1:]], dim=-1)
        return xy, yz, xz

    def encode_x(self, xyz, bound):
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)

    def encode_audio(self, a):
        if a is None:
            return None
        enc_a = self.audio_net(a)
        enc_a = self.audio_att_net(enc_a.unsqueeze(0))
        return enc_a

    def forward(self, x, a, e=None, c=None, phoneme_id=None, viseme_id=None, phoneme_support=None,
                effective_coverage=None):
        enc_x = self.encode_x(x, bound=self.bound)

        enc_a = self.encode_audio(a)
        enc_a = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a * aud_ch_att
        h = torch.cat([enc_x, enc_w], dim=-1)

        eye_att = None
        if self.exp_eye:
            eye_att = torch.relu(self.eye_att_net(enc_x))
            enc_e = self.exp_encode_net(e[:-1])
            enc_e = torch.cat([enc_e, e[-1:]], dim=-1)
            enc_e = enc_e * eye_att
            h = torch.cat([h, enc_e], dim=-1)

        if c is not None:
            c = c.repeat(enc_x.shape[0], 1)
            h = torch.cat([h, c], dim=-1)

        h = self.sigma_net(h)

        d_xyz = h[..., :3] * 1e-2
        d_rot = h[..., 3:7]
        if self.args.type == 'face':
            d_opa = h[..., 7:8]
            d_scale = h[..., 8:11]
        else:
            d_opa = None
            d_scale = None

        p = self.align_net(enc_x)
        p_xyz = p[..., :3] * 1e-2
        p_scale = torch.tanh(p[..., 3:] / 5) * 0.25 + 1

        # ============================================================
        # PPR baseline
        # ============================================================
        phoneme_d_xyz = torch.zeros_like(d_xyz)
        if self.use_phoneme_residual and phoneme_id is not None:
            if not torch.is_tensor(phoneme_id):
                phoneme_id = torch.tensor(phoneme_id, device=x.device, dtype=torch.long)
            phoneme_id = phoneme_id.to(device=x.device, dtype=torch.long).view(-1)[:1]
            phoneme_id = torch.clamp(phoneme_id, min=0, max=self.num_phonemes - 1)

            coeff = self.phoneme_coeff(phoneme_id)[0]
            basis = self.phoneme_basis_net(enc_x).view(-1, self.phoneme_rank, 3)
            phoneme_d_xyz = (basis * coeff.view(1, self.phoneme_rank, 1)).sum(dim=1)
            phoneme_d_xyz = phoneme_d_xyz * 1e-2 * self.phoneme_res_scale

        # ============================================================
        # CA-MLVSR
        # ============================================================
        viseme_d_xyz = torch.zeros_like(d_xyz)
        viseme_d_xyz_local = torch.zeros_like(d_xyz)
        viseme_gate = torch.zeros((d_xyz.shape[0], 1), device=d_xyz.device, dtype=d_xyz.dtype)
        coverage_scale = torch.ones((1, 1), device=x.device, dtype=x.dtype)
        local_confidence = torch.zeros((1, 1), device=x.device, dtype=x.dtype)
        global_confidence = torch.zeros((1, 1), device=x.device, dtype=x.dtype)

        if self.use_viseme_shared_residual and viseme_id is not None:
            if not torch.is_tensor(viseme_id):
                viseme_id = torch.tensor(viseme_id, device=x.device, dtype=torch.long)
            viseme_id = viseme_id.to(device=x.device, dtype=torch.long).view(-1)[:1]
            viseme_id = torch.clamp(viseme_id, min=0, max=self.num_visemes - 1)

            # Shared viseme prototype residual
            coeff = self.viseme_coeff(viseme_id)[0]
            basis = self.viseme_basis_net(enc_x).view(-1, self.viseme_rank, 3)
            viseme_d_xyz_raw = (basis * coeff.view(1, self.viseme_rank, 1)).sum(dim=1)

            # Audio-guided spatial gate
            audio_gate = aud_ch_att.norm(dim=-1, keepdim=True)
            gate_max = audio_gate.max().detach().clamp_min(1e-6)
            audio_gate = (audio_gate / gate_max).clamp(min=0.0, max=1.0).detach()
            viseme_gate = audio_gate.pow(self.viseme_gate_gamma)

            # Localized residual before coverage scaling.
            # Regularize this quantity, not the final scaled residual.
            viseme_d_xyz_local = viseme_d_xyz_raw * 1e-2 * self.viseme_res_scale * viseme_gate

            if self.use_coverage_adaptive and phoneme_support is not None and effective_coverage is not None:
                if not torch.is_tensor(phoneme_support):
                    phoneme_support = torch.tensor(phoneme_support, device=x.device, dtype=x.dtype)
                support = phoneme_support.to(device=x.device, dtype=x.dtype).view(1, 1).clamp_min(0.0)

                if not torch.is_tensor(effective_coverage):
                    effective_coverage = torch.tensor(effective_coverage, device=x.device, dtype=x.dtype)
                global_confidence = effective_coverage.to(device=x.device, dtype=x.dtype).view(1, 1).clamp(0.0, 1.0)

                tau = max(float(self.coverage_tau), 1e-6)
                local_confidence = support / (support + tau)
                evidence_confidence = global_confidence * local_confidence
                coverage_scale = (1.0 - evidence_confidence).clamp(0.0, 1.0)

            viseme_d_xyz = viseme_d_xyz_local * coverage_scale

        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud': aud_ch_att.norm(dim=-1, keepdim=True),
            'ambient_eye': eye_att.norm(dim=-1, keepdim=True) if eye_att is not None else None,
            'p_xyz': p_xyz,
            'p_scale': p_scale,
            'phoneme_d_xyz': phoneme_d_xyz,
            'viseme_d_xyz': viseme_d_xyz,
            'viseme_d_xyz_local': viseme_d_xyz_local,
            'viseme_gate': viseme_gate,
            'coverage_scale': coverage_scale,
            'local_confidence': local_confidence,
            'global_confidence': global_confidence,
        }

    def get_params(self, lr, lr_net, wd=0):
        params = [
            {'params': self.audio_net.parameters(), 'name': 'neural_audio_net', 'lr': lr_net, 'weight_decay': wd},
            {'params': self.encoder_xy.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'name': 'neural_encoder_xy', 'lr': lr},
            {'params': self.sigma_net.parameters(), 'name': 'neural_sigma_net', 'lr': lr_net, 'weight_decay': wd},
            {'params': self.align_net.parameters(), 'name': 'neural_align_net', 'lr': lr_net / 2, 'weight_decay': wd},
        ]

        params.append({'params': self.audio_att_net.parameters(), 'name': 'neural_audio_att_net', 'lr': lr_net * 5,
                       'weight_decay': 0.0001})
        if self.individual_dim > 0:
            params.append(
                {'params': self.individual_codes, 'name': 'neural_individual_codes', 'lr': lr_net, 'weight_decay': wd})

        params.append({'params': self.aud_ch_att_net.parameters(), 'name': 'neural_aud_ch_att_net', 'lr': lr_net,
                       'weight_decay': wd})

        if self.exp_eye:
            params.append({'params': self.eye_att_net.parameters(), 'name': 'neural_eye_att_net', 'lr': lr_net,
                           'weight_decay': wd})
            params.append({'params': self.exp_encode_net.parameters(), 'name': 'neural_exp_encode_net', 'lr': lr_net,
                           'weight_decay': wd})

        if self.use_phoneme_residual:
            params.append({'params': self.phoneme_coeff.parameters(), 'name': 'neural_phoneme_coeff', 'lr': lr_net * 5,
                           'weight_decay': 0.0})
            params.append({'params': self.phoneme_basis_net.parameters(), 'name': 'neural_phoneme_basis', 'lr': lr_net,
                           'weight_decay': wd})

        if self.use_viseme_shared_residual:
            params.append({'params': self.viseme_coeff.parameters(), 'name': 'neural_viseme_coeff', 'lr': lr_net * 5,
                           'weight_decay': 0.0})
            params.append({'params': self.viseme_basis_net.parameters(), 'name': 'neural_viseme_basis', 'lr': lr_net,
                           'weight_decay': wd})

        return params
