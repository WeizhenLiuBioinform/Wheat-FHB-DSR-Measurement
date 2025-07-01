import torch
import torch.nn as nn
# from ultralytics.utils.torch_utils import fuse_conv_and_bn
 
__all__ = ("RepVGGDW", "C2fCIB",  "PSA", "SCDown")
 
def fuse_conv_and_bn(conv, bn):
    """Fuse Conv2d() and BatchNorm2d() layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/."""
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )
 
    # Prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))
 
    # Prepare spatial bias
    b_conv = torch.zeros(conv.weight.shape[0], device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)
 
    return fusedconv
    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
 
class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
 
    default_act = nn.SiLU()  # default activation
 
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
 
    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))
 
    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
 
class Bottleneck(nn.Module):
    """Standard bottleneck."""
 
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
 
 
class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.conv(x) + self.conv1(x))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))
 
    @torch.no_grad()
    def fuse(self):
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [2,2,2,2])
 
        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b
 
        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)
 
        self.conv = conv
        del self.conv1
 
class CIB(nn.Module):
    """Standard bottleneck."""
 
    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            Conv(2 * c_, 2 * c_, 3, g=2 * c_) if not lk else RepVGGDW(2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )
 
        self.add = shortcut and c1 == c2
 
    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv1(x) if self.add else self.cv1(x)
       
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
 
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
 
 
class C2fCIB(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""
 
    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))
 
 
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8,
                 attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
 
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
 
        attn = (
            (q.transpose(-2, -1) @ k) * self.scale
        )
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSA(nn.Module):
 
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))

 
class SA_GELAN_Block(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, c1, 1, 1)  # Conv layer for splitting
        self.cv1_2 = Conv(self.c, self.c, 1, 1)
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.cv2 = Conv(2 * self.c, c2, 1, 1)  # Conv layer for final integration
        self.cv3 = Conv(self.c, self.c, 5, 1, p=2)

    def forward(self, x):
        # Step 1: Apply cv1 and split into c and d
        c, d = self.cv1(x).split((self.c, self.c), dim=1)
        # Step 2: Process c through attention and add back to c
        k = c + self.attn(c)  # Ensure k has the same channel size as c
        # Step 3: Process d through more splitting and pooling
        e, f = self.cv1_2(d).split((self.c//2, self.c//2), dim=1)
        e = nn.AdaptiveAvgPool2d((e.size(2), e.size(3)))(e)
        f = nn.AdaptiveMaxPool2d((f.size(2), f.size(3)))(f)
        g = torch.cat([e, f], dim=1)
        # Step 4: Process g to get attention map h
        h = self.cv3(g)
        i = torch.sigmoid(h)  # Weight map
        # Step 5: Apply the attention map to g to get j
        j = g * i
        # Step 6: Combine k and j
        # Step 7: Apply cv2 to integrate features and get final output
        return self.cv2(torch.cat([k, j], dim=1))

class SA_GELAN(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Conv layer for splitting
        self.block1 = SA_GELAN_Block(self.c, self.c, e)
        self.block2 = SA_GELAN_Block(self.c, self.c, e)
        self.cv2 = Conv(3 * self.c, c1, 1, 1)  # Conv layer for final integration

    def forward(self, x):
        # Step 1: Initial split into a and b
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        # Step 2: Pass b through the first block
        m = self.block1(b)
        # Step 3: Pass m through the second block
        n = self.block2(m)
        # Step 4: Concatenate a, m, and n, then final cv2
        return self.cv2(torch.cat([a, m, n], dim=1))


class SA_GELAN_Block2(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, c1, 1, 1)  # Conv layer for splitting
        self.cv1_2 = Conv(self.c, self.c, 1, 1)
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.cv2 = Conv(2 * self.c, c2, 1, 1)  # Conv layer for final integration
        self.cv3 = Conv(self.c, self.c, 5, 1, p=2)
        self.residual_conv = Conv(c1, c2, 1, 1)  # Conv layer for residual connection

    def forward(self, x):
        # Step 1: Apply cv1 and split into c and d
        c, d = self.cv1(x).split((self.c, self.c), dim=1)
        # Step 2: Process c through attention and add back to c
        k = c + self.attn(c)  # Ensure k has the same channel size as c
        # Step 3: Process d through more splitting and pooling
        e, f = self.cv1_2(d).split((self.c//2, self.c//2), dim=1)
        e = nn.AdaptiveAvgPool2d((e.size(2), e.size(3)))(e)
        f = nn.AdaptiveMaxPool2d((f.size(2), f.size(3)))(f)
        g = torch.cat([e, f], dim=1)
        # Step 4: Process g to get attention map h
        h = self.cv3(g)
        i = torch.sigmoid(h)  # Weight map
        # Step 5: Apply the attention map to g to get j
        j = g * i
        # Step 6: Combine k and j
        combined = torch.cat([k, j], dim=1)
        # Step 7: Apply cv2 to integrate features and get final output
        output = self.cv2(combined)
        # Step 8: Add residual connection
        output += self.residual_conv(x)
        return output

class SA_GELAN2(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Conv layer for splitting
        self.block1 = SA_GELAN_Block2(self.c, self.c, e)
        self.block2 = SA_GELAN_Block2(self.c, self.c, e)
        self.cv2 = Conv(3 * self.c, c1, 1, 1)  # Conv layer for final integration
        self.residual_conv = Conv(c1, c1, 1, 1)  # Conv layer for residual connection

    def forward(self, x):
        # Step 1: Initial split into a and b
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        # Step 2: Pass b through the first block
        m = self.block1(b)
        # Step 3: Pass m through the second block
        n = self.block2(m)
        # Step 4: Concatenate a, m, and n, then final cv2
        combined = torch.cat([a, m, n], dim=1)
        output = self.cv2(combined)
        # Step 5: Add residual connection
        output += self.residual_conv(x)
        return output

class SA_GELAN3(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # Conv layer for splitting
        self.block1 = PSA(self.c, self.c, e)   # Replace with PSA
        self.block2 = PSA(self.c, self.c, e)   # Replace with PSA
        self.cv2 = Conv(3 * self.c, c1, 1, 1)  # Conv layer for final integration
        self.residual_conv = Conv(c1, c1, 1, 1)  # Conv layer for residual connection

    def forward(self, x):
        # Step 1: Initial split into a and b
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        # Step 2: Pass b through the first block (now PSA)
        m = self.block1(b)
        # Step 3: Pass m through the second block (now PSA)
        n = self.block2(m)
        # Step 4: Concatenate a, m, and n, then final cv2
        combined = torch.cat([a, m, n], dim=1)
        output = self.cv2(combined)
        # Step 5: Add residual connection
        output += self.residual_conv(x)
        return output
    


import torch
import torch.nn as nn
import torch.nn.functional as F

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.haar_weights = nn.Conv2d(1, 4, kernel_size=2, stride=2, padding=0, bias=False, groups=1)
        self.haar_weights.weight.data = torch.tensor([[[[1, 1], [1, 1]]],   # 低频
                                                      [[[1, 1], [-1, -1]]],  # 高频水平
                                                      [[[1, -1], [1, -1]]],  # 高频垂直
                                                      [[[1, -1], [-1, 1]]]], # 高频对角
                                                     dtype=torch.float32) / 2.0

    def forward(self, x):
        B, C, H, W = x.shape
        haar_weights = self.haar_weights.weight.repeat(C, 1, 1, 1).to(x.device)
        # 使用 nn.Conv2d 替代 F.conv2d
        conv = nn.Conv2d(C, C * 4, kernel_size=2, stride=2, padding=0, bias=False, groups=C)
        conv.weight.data = haar_weights
        return conv(x)

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.haar_weights = nn.ConvTranspose2d(4, 1, kernel_size=2, stride=2, padding=0, bias=False, groups=1)
        self.haar_weights.weight.data = torch.tensor([[[[1, 1], [1, 1]]],   # 低频
                                                      [[[1, 1], [-1, -1]]],  # 高频水平
                                                      [[[1, -1], [1, -1]]],  # 高频垂直
                                                      [[[1, -1], [-1, 1]]]], # 高频对角
                                                     dtype=torch.float32) / 2.0

    def forward(self, x, output_size):
        B, C, H, W = x.shape
        haar_weights = self.haar_weights.weight.repeat(C // 4, 1, 1, 1).to(x.device)
        conv_trans = nn.ConvTranspose2d(C, C // 4, kernel_size=2, stride=2, padding=0, bias=False, groups=C // 4)
        conv_trans.weight.data = haar_weights
        x = conv_trans(x)
        return F.interpolate(x, size=output_size, mode='nearest')


class SA_GELAN4(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1)
        self.block1 = PSA(self.c, self.c, e)
        self.block2 = PSA(self.c, self.c, e)
        self.cv2 = nn.Conv2d(3 * self.c, c1, kernel_size=1, stride=1)
        self.residual_conv = nn.Conv2d(c1, c1, kernel_size=1, stride=1)
        
        # DWT and IDWT
        self.dwt = DWT()
        self.idwt = IDWT()

        # 潜在空间的注意力机制
        self.latent_attention = nn.Sequential(
            nn.Conv2d(4 * self.c, 4 * self.c, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(4 * self.c, 4 * self.c, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Step 1: Initial split into a and b
        a, b = self.cv1(x).split((self.c, self.c), dim=1)

        # 保存原始特征图的空间尺寸
        original_size = b.shape[2:]

        # Step 2: 将 b 通过 DWT 转换到潜在空间
        dwt_b = self.dwt(b)

        # Step 3: 在潜在空间中进行注意力加权
        attn_map = self.latent_attention(dwt_b)
        dwt_b_weighted = dwt_b * attn_map

        # Step 4: 逆 DWT 将特征映射回原空间，确保尺寸与原始一致
        b_reconstructed = self.idwt(dwt_b_weighted, output_size=original_size)

        # Step 5: Pass b_reconstructed through the first block (now PSA)
        m = self.block1(b_reconstructed)

        # Step 6: Pass m through the second block (now PSA)
        n = self.block2(m)

        # Step 7: Concatenate a, m, and n, then final cv2
        combined = torch.cat([a, m, n], dim=1)
        output = self.cv2(combined)

        # Step 8: Add residual connection
        output += self.residual_conv(x)
        return output

 
import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

    def forward(self, x):
        batch_size, C, H, W = x.size()
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        attention_map = torch.bmm(theta_x.permute(0, 2, 1), phi_x)
        attention_map = F.softmax(attention_map, dim=-1)

        y = torch.bmm(g_x, attention_map.permute(0, 2, 1))
        y = y.view(batch_size, self.inter_channels, H, W)
        y = self.W(y)
        return y + x  # 残差连接

class DWT(nn.Module):
    # 保持原先的多尺度小波分解
    def __init__(self):
        super(DWT, self).__init__()
        self.haar_weights = nn.Parameter(
            torch.tensor([[[[1, 1], [1, 1]]],   # 低频
                          [[[1, 1], [-1, -1]]],  # 高频水平
                          [[[1, -1], [1, -1]]],  # 高频垂直
                          [[[1, -1], [-1, 1]]]], # 高频对角
                         dtype=torch.float32) / 2.0,
            requires_grad=False
        )
        self.conv = None

    def forward(self, x):
        B, C, H, W = x.shape
        if self.conv is None:
            self.conv = nn.Conv2d(C, C * 4, kernel_size=2, stride=2, padding=0, bias=False, groups=C).to(x.device)
            self.conv.weight.data = self.haar_weights.repeat(C, 1, 1, 1).to(x.device)
        return self.conv(x)

class IDWT(nn.Module):
    def __init__(self):
        super(IDWT, self).__init__()
        self.haar_weights = nn.Parameter(
            torch.tensor([[[[1, 1], [1, 1]]],   # 低频
                          [[[1, 1], [-1, -1]]],  # 高频水平
                          [[[1, -1], [1, -1]]],  # 高频垂直
                          [[[1, -1], [-1, 1]]]], # 高频对角
                         dtype=torch.float32) / 2.0,
            requires_grad=False
        )
        self.conv_trans = None

    def forward(self, x, output_size):
        B, C, H, W = x.shape
        if self.conv_trans is None:
            self.conv_trans = nn.ConvTranspose2d(C, C // 4, kernel_size=2, stride=2, padding=0, bias=False, groups=C // 4).to(x.device)
            self.conv_trans.weight.data = self.haar_weights.repeat(C // 4, 1, 1, 1).to(x.device)
        x = self.conv_trans(x)
        return F.interpolate(x, size=output_size, mode='bilinear', align_corners=True)

class SA_GELAN5_Optimized(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1)
        self.block1 = PSA(self.c, self.c, e)
        self.block2 = PSA(self.c, self.c, e)
        self.cv2 = nn.Conv2d(3 * self.c, c1, kernel_size=1, stride=1)
        self.residual_conv = nn.Conv2d(c1, c1, kernel_size=1, stride=1)
        
        # DWT and IDWT
        self.dwt = DWT()
        self.idwt = IDWT()

        # 引入非局部操作和混合注意力机制
        self.non_local_block = NonLocalBlock(4 * self.c)

        # 潜在空间的注意力机制 - 混合空间和频域
        self.latent_attention = nn.Sequential(
            nn.Conv2d(4 * self.c, 4 * self.c, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(4 * self.c, 4 * self.c, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Step 1: Initial split into a and b
        a, b = self.cv1(x).split((self.c, self.c), dim=1)

        # 保存原始特征图的空间尺寸
        original_size = b.shape[2:]

        # Step 2: 将 b 通过 DWT 转换到潜在空间
        dwt_b = self.dwt(b)

        # Step 3: 在潜在空间中使用非局部操作进行全局上下文信息获取
        non_local_dwt_b = self.non_local_block(dwt_b)

        # Step 4: 在潜在空间中进行混合注意力加权
        attn_map = self.latent_attention(non_local_dwt_b)
        dwt_b_weighted = non_local_dwt_b * attn_map

        # Step 5: 逆 DWT 将特征映射回原空间，确保尺寸与原始一致
        b_reconstructed = self.idwt(dwt_b_weighted, output_size=original_size)

        # Step 6: Pass b_reconstructed through the first block (now PSA)
        m = self.block1(b_reconstructed)

        # Step 7: Pass m through the second block (now PSA)
        n = self.block2(m)

        # Step 8: Concatenate a, m, and n, then final cv2
        combined = torch.cat([a, m, n], dim=1)
        output = self.cv2(combined)

        # Step 9: Add residual connection
        output += self.residual_conv(x)
        return output


import torch
import torch.nn as nn
import torch.fft
from torch.cuda.amp import autocast

class FFTAttentionModule(nn.Module):
    def __init__(self, c):
        super(FFTAttentionModule, self).__init__()
        self.latent_attention = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(c, c, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Step 1: Convert to float32 before applying FFT
        x = x.float()

        # Step 2: Apply FFT to map features to frequency domain
        fft_x = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')

        # Step 3: Separate real and imaginary parts
        real, imag = fft_x.real, fft_x.imag

        # Step 4: Convert real part back to the same dtype as latent_attention weights
        real = real.to(self.latent_attention[0].weight.dtype)

        # Step 5: Apply attention in the frequency domain (on the real part for simplicity)
        attn_map = self.latent_attention(real)
        real_weighted = real * attn_map

        # Step 6: Convert real_weighted and imag to float32 for complex combination
        real_weighted = real_weighted.float()
        imag = imag.float()

        # Step 7: Recombine real and imaginary parts
        fft_x_weighted = torch.complex(real_weighted, imag)

        # Step 8: Apply inverse FFT to map features back to spatial domain
        ifft_x = torch.fft.ifftn(fft_x_weighted, dim=(-2, -1), norm='ortho')

        # Return only the real part of the inverse FFT result, and convert back to the original type
        return ifft_x.real.to(x.dtype)

class SA_GELAN_FFT(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super(SA_GELAN_FFT, self).__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1)
        self.block1 = PSA(self.c, self.c, e)
        self.block2 = PSA(self.c, self.c, e)
        self.cv2 = nn.Conv2d(3 * self.c, c1, kernel_size=1, stride=1)
        self.residual_conv = nn.Conv2d(c1, c1, kernel_size=1, stride=1)

        # FFT attention module instead of DWT/IDWT
        self.fft_attention = FFTAttentionModule(self.c)

    def forward(self, x):
        with autocast():
        # Step 1: Initial split into a and b
            a, b = self.cv1(x).split((self.c, self.c), dim=1)

        # Step 2: Apply FFT-based attention to b
            b_weighted = self.fft_attention(b)

        # Step 3: Pass b_weighted through the first block (now PSA)
            m = self.block1(b_weighted)

        # Step 4: Pass m through the second block (now PSA)
            n = self.block2(m)

        # Step 5: Concatenate a, m, and n, then final cv2
            combined = torch.cat([a, m, n], dim=1)
            output = self.cv2(combined)

        # Step 6: Add residual connection
            output += self.residual_conv(x)
        return output


import torch
import torch.nn as nn
import torch.fft
from torch.cuda.amp import autocast

class MultiScaleFFTAttentionModule(nn.Module):
    def __init__(self, c):
        super(MultiScaleFFTAttentionModule, self).__init__()
        self.latent_attention = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=1),
            nn.Sigmoid()
        )

        # 固定的下采样和上采样层
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        # 保存原始数据类型
        original_dtype = x.dtype

        # 尺度1（原始尺度）
        x1 = x.float()
        fft_x1 = torch.fft.fftn(x1, dim=(-2, -1), norm='ortho')
        real1, imag1 = fft_x1.real, fft_x1.imag
        real1 = real1.to(self.latent_attention[0].weight.dtype)
        attn_map1 = self.latent_attention(real1)
        real_weighted1 = real1 * attn_map1
        fft_x_weighted1 = torch.complex(real_weighted1.float(), imag1.float())
        ifft_x1 = torch.fft.ifftn(fft_x_weighted1, dim=(-2, -1), norm='ortho')
        out1 = ifft_x1.real.to(original_dtype)

        # 尺度2（下采样后）
        x2 = self.downsample(x).float()
        fft_x2 = torch.fft.fftn(x2, dim=(-2, -1), norm='ortho')
        real2, imag2 = fft_x2.real, fft_x2.imag
        real2 = real2.to(self.latent_attention[0].weight.dtype)
        attn_map2 = self.latent_attention(real2)
        real_weighted2 = real2 * attn_map2
        fft_x_weighted2 = torch.complex(real_weighted2.float(), imag2.float())
        ifft_x2 = torch.fft.ifftn(fft_x_weighted2, dim=(-2, -1), norm='ortho')
        out2 = ifft_x2.real.to(original_dtype)
        out2 = self.upsample(out2)

        # 确保输出尺寸一致
        assert out1.shape == out2.shape, f"Output shapes are not equal: {out1.shape}, {out2.shape}"

        # 将特征相加
        combined_features = out1 + out2

        return combined_features


class SA_GELAN_FFT3(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super(SA_GELAN_FFT3, self).__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1)
        self.block1 = PSA(self.c, self.c, e)
        self.block2 = PSA(self.c, self.c, e)
        self.cv2 = nn.Conv2d(3 * self.c, c1, kernel_size=1, stride=1)
        self.residual_conv = nn.Conv2d(c1, c1, kernel_size=1, stride=1)

        # 使用修改后的 MultiScaleFFTAttentionModule
        self.fft_attention = MultiScaleFFTAttentionModule(self.c)

    def forward(self, x):
        with autocast():
            # 初始分割为 a 和 b
            a, b = self.cv1(x).split((self.c, self.c), dim=1)

            # 对 b 应用多尺度 FFT 注意力
            b_weighted = self.fft_attention(b)

            # 通过两个 PSA 块
            m = self.block1(b_weighted)
            n = self.block2(m)

            # 拼接并经过最终的卷积
            combined = torch.cat([a, m, n], dim=1)
            output = self.cv2(combined)

            # 添加残差连接
            output += self.residual_conv(x)
        return output

# class MultiScaleFFTAttentionModule(nn.Module):
#     def __init__(self, c, num_scales=3):
#         super(MultiScaleFFTAttentionModule, self).__init__()
#         self.num_scales = num_scales
#         self.latent_attention = nn.Sequential(
#             nn.Conv2d(c, c, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(c, c, kernel_size=1),
#             nn.Sigmoid()
#         )

#     def apply_fft(self, x):
#         # Step 1: Convert to float32 before applying FFT
#         x = x.float()

#         # Step 2: Apply FFT to map features to frequency domain
#         fft_x = torch.fft.fftn(x, dim=(-2, -1), norm='ortho')

#         # Step 3: Separate real and imaginary parts
#         real, imag = fft_x.real, fft_x.imag

#         return real, imag

#     def apply_ifft(self, real, imag, original_dtype):
#     # 确保 real 和 imag 都是相同的数据类型
#         real = real.float()
#         imag = imag.float()

#     # Step 7: Recombine real and imaginary parts
#         fft_x_weighted = torch.complex(real, imag)

#     # Step 8: Apply inverse FFT to map features back to spatial domain
#         ifft_x = torch.fft.ifftn(fft_x_weighted, dim=(-2, -1), norm='ortho')

#     # Return only the real part of the inverse FFT result
#         return ifft_x.real.to(original_dtype)


#     def forward(self, x):
#         original_dtype = x.dtype
#         scale_features = []

#         # Step 1: Process input at different scales
#         for scale in range(self.num_scales):
#             scale_factor = 1 / (2 ** scale)  # 每次下采样的比例
#             downsampled_x = F.interpolate(x, scale_factor=scale_factor, mode='bilinear', align_corners=False)
            
#             # Step 2: Apply FFT on the downsampled feature
#             real, imag = self.apply_fft(downsampled_x)

#             # Step 3: Apply attention in the frequency domain (on the real part)
#             real = real.to(self.latent_attention[0].weight.dtype)
#             attn_map = self.latent_attention(real)
#             real_weighted = real * attn_map

#             # Step 4: Recombine real and imaginary parts, and apply inverse FFT
#             ifft_x = self.apply_ifft(real_weighted, imag, original_dtype)

#             # Step 5: Upsample to original size
#             upsampled_x = F.interpolate(ifft_x, size=x.shape[-2:], mode='bilinear', align_corners=False)
#             scale_features.append(upsampled_x)

#         # Step 6: Sum features from all scales
#         combined_features = sum(scale_features)

#         return combined_features

# class SA_GELAN_FFT2(nn.Module):
#     def __init__(self, c1, c2, e=0.5, num_scales=3):
#         super(SA_GELAN_FFT2, self).__init__()
#         assert c1 == c2
#         self.c = int(c1 * e)
#         self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1)
#         self.block1 = PSA(self.c, self.c, e)
#         self.block2 = PSA(self.c, self.c, e)
#         self.cv2 = nn.Conv2d(3 * self.c, c1, kernel_size=1, stride=1)
#         self.residual_conv = nn.Conv2d(c1, c1, kernel_size=1, stride=1)

#         # Multi-scale FFT attention module
#         self.fft_attention = MultiScaleFFTAttentionModule(self.c, num_scales=num_scales)

#     def forward(self, x):
#         with autocast():
#             # Step 1: Initial split into a and b
#             a, b = self.cv1(x).split((self.c, self.c), dim=1)

#             # Step 2: Apply multi-scale FFT-based attention to b
#             b_weighted = self.fft_attention(b)

#             # Step 3: Pass b_weighted through the first block (now PSA)
#             m = self.block1(b_weighted)

#             # Step 4: Pass m through the second block (now PSA)
#             n = self.block2(m)

#             # Step 5: Concatenate a, m, and n, then final cv2
#             combined = torch.cat([a, m, n], dim=1)
#             output = self.cv2(combined)

#             # Step 6: Add residual connection
#             output += self.residual_conv(x)
#         return output


class MultiScaleFFTAttentionModule(nn.Module):
    def __init__(self, c):
        super(MultiScaleFFTAttentionModule, self).__init__()
        # 调整padding以匹配不同的dilation
        self.conv_scale1 = nn.Conv2d(c, c, kernel_size=3, padding=1, dilation=1)
        self.conv_scale2 = nn.Conv2d(c, c, kernel_size=3, padding=2, dilation=2)
        self.conv_scale3 = nn.Conv2d(c, c, kernel_size=3, padding=3, dilation=3)
        # 融合层
        self.fusion_conv = nn.Conv2d(c * 3, c, kernel_size=1)
        # 注意力层
        self.attention_conv = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, c, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 多尺度特征提取
        x1 = self.conv_scale1(x)
        x2 = self.conv_scale2(x)
        x3 = self.conv_scale3(x)

        # 确保输出尺寸一致
        assert x1.shape == x2.shape == x3.shape, f"Output shapes are not equal: {x1.shape}, {x2.shape}, {x3.shape}"

        # 将特征进行拼接
        multi_scale_features = torch.cat([x1, x2, x3], dim=1)

        # 融合特征
        fused_features = self.fusion_conv(multi_scale_features)

        # 应用注意力机制
        attn_map = self.attention_conv(fused_features)
        out = x * attn_map

        return out


class SA_GELAN_FFT2(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super(SA_GELAN_FFT2, self).__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1)
        self.block1 = PSA(self.c, self.c, e)
        self.block2 = PSA(self.c, self.c, e)
        self.cv2 = nn.Conv2d(3 * self.c, c1, kernel_size=1, stride=1)
        self.residual_conv = nn.Conv2d(c1, c1, kernel_size=1, stride=1)

        # 使用修改后的 MultiScaleFFTAttentionModule
        self.attention_module = MultiScaleFFTAttentionModule(self.c)

    def forward(self, x):
        with autocast():
            # 初始分割为 a 和 b
            a, b = self.cv1(x).split((self.c, self.c), dim=1)

            # 对 b 应用注意力模块
            b_weighted = self.attention_module(b)

            # 通过两个 PSA 块
            m = self.block1(b_weighted)
            n = self.block2(m)

            # 拼接并经过最终的卷积
            combined = torch.cat([a, m, n], dim=1)
            output = self.cv2(combined)

            # 添加残差连接
            output += self.residual_conv(x)
        return output


class SCDown(nn.Module):
    def __init__(self, c1, c2, k, s):
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)
 
    def forward(self, x):
        return self.cv2(self.cv1(x))


import torch
import torch.nn as nn

class DiffusionAttention(nn.Module):
    def __init__(self, channels, time_steps=10, alpha=0.1, epsilon=1e-5):
        super().__init__()
        self.channels = channels
        self.time_steps = time_steps
        self.alpha = alpha  # 平滑因子
        self.epsilon = epsilon  # 数值稳定性
        self.denoise_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        T = self.time_steps
        alphas = torch.linspace(0.9, 0.1, T).to(x.device)
        alpha_bars = torch.cumprod(alphas, dim=0).to(x.device)
        
        noise_std = 1.0  # 初始噪声强度
        noise = torch.randn_like(x)
        x_t = torch.sqrt(alpha_bars[-1] + self.epsilon) * x + torch.sqrt(1 - alpha_bars[-1] + self.epsilon) * noise
        weight_map_list = []
        kl_divergence_list = []
        
        prev_kl_divergence = None
        
        for t in range(T):
            predicted_noise = self.denoise_net(x_t)
            x_hat = (x_t - torch.sqrt(1 - alphas[t] + self.epsilon) * predicted_noise) / torch.sqrt(alpha_bars[t] + self.epsilon)
            
            # 使用均值和方差计算 KL 散度，添加 epsilon 以防止数值问题
            mu_P = x.mean(dim=(0, 2, 3))
            mu_Q = x_hat.mean(dim=(0, 2, 3))
            sigma_P = x.var(dim=(0, 2, 3)) + self.epsilon
            sigma_Q = x_hat.var(dim=(0, 2, 3)) + self.epsilon

            kl_divergence = 0.5 * (torch.log(sigma_Q / sigma_P) + 
                                   (sigma_P / sigma_Q) + 
                                   (mu_P - mu_Q)**2 / sigma_Q - 
                                   1).sum()
            
            # 调整噪声强度
            if prev_kl_divergence is not None:
                kl_diff = kl_divergence - prev_kl_divergence
                adjustment = self.alpha * (-kl_diff).clamp(-0.1, 0.1)  # 使用平滑的调整策略
                noise_std *= (1 + adjustment)
            
            prev_kl_divergence = kl_divergence
            
            kl_divergence_list.append(kl_divergence)
            weight_map_list.append(predicted_noise)
        
        # 选择与原始特征图散度最小的时间步所对应的权重图
        min_kl_index = torch.argmin(torch.stack(kl_divergence_list))
        best_weight_map = weight_map_list[min_kl_index]
        
        weighted_average_weight_map = torch.sigmoid(best_weight_map)
        return weighted_average_weight_map

class Diffusion_PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5, attn_ratio=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=attn_ratio, num_heads=self.c // 64)
        self.diffusion_attn = DiffusionAttention(self.c, time_steps=10, alpha=0.1)  # 传入平滑因子
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
        # 生成扩散过程的权重图
        diffusion_weight_map = self.diffusion_attn(b)
    
        # 在自注意力机制之前应用权重图
        b = b * diffusion_weight_map
    
        # 原始自注意力机制
        original_attn = self.attn(b)
    
        b = b + original_attn
        b = b + self.ffn(b)
        
        return self.cv2(torch.cat((a, b), 1))



#直接加注意力上
# class DiffusionAttention5(nn.Module):
#     def __init__(self, channels, time_steps=10, alpha=0.1):
#         super().__init__()
#         self.channels = channels
#         self.time_steps = time_steps
#         self.alpha = alpha  # 平滑因子
#         self.denoise_net = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         )
        
#     def forward(self, x):
#         T = self.time_steps
#         alphas = torch.linspace(0.9, 0.1, T).to(x.device)
#         alpha_bars = torch.cumprod(alphas, dim=0).to(x.device)
        
#         noise_std = 1.0  # 初始噪声强度
#         noise = torch.randn_like(x)
#         x_t = torch.sqrt(alpha_bars[-1]) * x + torch.sqrt(1 - alpha_bars[-1]) * noise
#         weight_map_list = []
#         kl_divergence_list = []
        
#         for t in range(T):
#             predicted_noise = self.denoise_net(x_t)
#             x_hat = (x_t - torch.sqrt(1 - alphas[t]) * predicted_noise) / torch.sqrt(alpha_bars[t])
            
#             # 使用均值和方差计算 KL 散度
#             mu_P = x.mean(dim=(0, 2, 3))
#             mu_Q = x_hat.mean(dim=(0, 2, 3))
#             sigma_P = x.var(dim=(0, 2, 3))
#             sigma_Q = x_hat.var(dim=(0, 2, 3))

#             kl_divergence = 0.5 * (torch.log(sigma_Q / sigma_P) + 
#                                    (sigma_P / sigma_Q) + 
#                                    (mu_P - mu_Q)**2 / sigma_Q - 
#                                    1).sum()
            
#             # 平滑调整策略
#             adjustment = self.alpha * (1.0 if kl_divergence < 1.0 else -1.0)
#             noise_std *= (1 + adjustment)  # 根据平滑因子调整噪声强度
            
#             kl_divergence_list.append(kl_divergence)
#             weight_map_list.append(predicted_noise)
        
#         # 计算加权平均权重图
#         weights = [1.0 / kl for kl in kl_divergence_list]
#         weights = torch.tensor(weights).to(x.device) / sum(weights)
        
#         weighted_average_weight_map = torch.zeros_like(weight_map_list[0])
#         for i, weight_map in enumerate(weight_map_list):
#             weighted_average_weight_map += weights[i] * weight_map
        
#         weighted_average_weight_map = torch.sigmoid(weighted_average_weight_map)
#         return weighted_average_weight_map

# class Attention5(nn.Module):
#     def __init__(self, dim, num_heads=8, attn_ratio=0.5):
#         super().__init__()
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.key_dim = int(self.head_dim * attn_ratio)
#         self.scale = self.key_dim ** -0.5
#         nh_kd = self.key_dim * num_heads
#         h = dim + nh_kd * 2
#         self.qkv = Conv(dim, h, 1, act=False)
#         self.proj = Conv(dim, dim, 1, act=False)
#         self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)
#         self.diffusion_attn = DiffusionAttention5(dim, time_steps=10, alpha=0.1)
 
#     def forward(self, x):
#         B, C, H, W = x.shape
#         N = H * W
#         qkv = self.qkv(x)
#         q, k, v = qkv.view(B, self.num_heads, self.key_dim*2 + self.head_dim, N).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
 
#         attn = (q.transpose(-2, -1) @ k) * self.scale
#         attn = attn.softmax(dim=-1)

#         # 使用 DiffusionAttention 生成的权重图进行加权
#         diffusion_weight_map = self.diffusion_attn(x)
#         attn = attn * diffusion_weight_map.view(B, 1, N, N)  # 扩展维度以适应attn的形状

#         x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
#         x = self.proj(x)
#         return x

# class Diffusion_PSA5(nn.Module):
#     def __init__(self, c1, c2, e=0.5, attn_ratio=0.5):
#         super().__init__()
#         assert(c1 == c2)
#         self.c = int(c1 * e)
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv(2 * self.c, c1, 1)
        
#         self.attn = Attention5(self.c, attn_ratio=attn_ratio, num_heads=self.c // 64)
        
#     def forward(self, x):
#         a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
#         # 应用自注意力机制，包括扩散过程生成的权重图
#         b = b + self.attn(b)
        
#         return self.cv2(torch.cat((a, b), 1))

#基于kde和kl综合评估-------------------version1
# import torch
# import torch.nn as nn
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from scipy.stats import gaussian_kde
# import numpy as np

# class DiffusionAttention2(nn.Module):
#     def __init__(self, channels, time_steps=10, alpha=0.1, epsilon=1e-5, kl_weight=0.5, pca_components=5, regularization=1e-5):
#         super().__init__()
#         self.channels = channels
#         self.time_steps = time_steps
#         self.alpha = alpha  # 平滑因子
#         self.epsilon = epsilon  # 数值稳定性
#         self.kl_weight = kl_weight  # KDE KL 散度的权重
#         self.pca_components = pca_components  # PCA 降维后的组件数
#         self.regularization = regularization  # 协方差矩阵正则化项
#         self.denoise_net = nn.Sequential(
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         )

#     def traditional_kl_divergence(self, x, x_hat):
#         mu_P = x.mean(dim=(0, 2, 3))
#         mu_Q = x_hat.mean(dim=(0, 2, 3))
#         sigma_P = x.var(dim=(0, 2, 3)) + self.epsilon
#         sigma_Q = x_hat.var(dim=(0, 2, 3)) + self.epsilon

#         kl_divergence = 0.5 * (torch.log(sigma_Q / sigma_P) + 
#                                (sigma_P / sigma_Q) + 
#                                (mu_P - mu_Q)**2 / sigma_Q - 
#                                1).sum()
#         return kl_divergence
    
#     def kde_kl_divergence(self, x, x_hat):
#         scaler = StandardScaler()
#         x_np = scaler.fit_transform(x.flatten().detach().cpu().numpy().reshape(-1, 1))
#         x_hat_np = scaler.transform(x_hat.flatten().detach().cpu().numpy().reshape(-1, 1))

#         # 动态调整 n_components 确保其合法性
#         pca_components = min(self.pca_components, x_np.shape[0], x_np.shape[1], x_hat_np.shape[0], x_hat_np.shape[1])
#         pca = PCA(n_components=pca_components)

#         try:
#             x_pca = pca.fit_transform(x_np)
#             x_hat_pca = pca.transform(x_hat_np)
#         except ValueError as e:
#             # 如果 PCA 失败，直接返回零 KL 散度
#             return torch.tensor(0.0, device=x.device)

#         # 处理总方差接近零的情况，防止 invalid divide
#         total_var = pca.explained_variance_.sum() + self.epsilon
#         explained_variance_ratio_ = pca.explained_variance_ / total_var

#         if np.isnan(explained_variance_ratio_).any():
#             return torch.tensor(0.0, device=x.device)
        
#         if x_pca.shape[1] <= 1:
#             return torch.tensor(0.0, device=x.device)

#         kde_p = gaussian_kde(x_pca.T, bw_method='silverman')
#         kde_p.covariance += np.eye(kde_p.covariance.shape[0]) * self.regularization

#         kde_q = gaussian_kde(x_hat_pca.T, bw_method='silverman')
#         kde_q.covariance += np.eye(kde_q.covariance.shape[0]) * self.regularization

#         p_vals = kde_p(x_pca.T) + self.epsilon
#         q_vals = kde_q(x_hat_pca.T) + self.epsilon
        
#         kl_divergence = np.sum(p_vals * np.log(p_vals / q_vals))
#         return torch.tensor(kl_divergence, device=x.device)
    
#     def combined_kl_divergence(self, x, x_hat):
#         traditional_kl = self.traditional_kl_divergence(x, x_hat)
#         kde_kl = self.kde_kl_divergence(x, x_hat)
        
#         combined_kl = (1 - self.kl_weight) * traditional_kl + self.kl_weight * kde_kl
#         return combined_kl
        
#     def forward(self, x):
#         T = self.time_steps
#         alphas = torch.linspace(0.9, 0.1, T).to(x.device)
#         alpha_bars = torch.cumprod(alphas, dim=0).to(x.device)
        
#         noise_std = 1.0
#         noise = torch.randn_like(x)
#         x_t = torch.sqrt(alpha_bars[-1] + self.epsilon) * x + torch.sqrt(1 - alpha_bars[-1] + self.epsilon) * noise
#         weight_map_list = []
#         kl_divergence_list = []

#         prev_kl_divergence = None
        
#         for t in range(T):
#             predicted_noise = self.denoise_net(x_t)
#             x_hat = (x_t - torch.sqrt(1 - alphas[t] + self.epsilon) * predicted_noise) / torch.sqrt(alpha_bars[t] + self.epsilon)
            
#             kl_divergence = self.combined_kl_divergence(x, x_hat)

#             if prev_kl_divergence is not None:
#                 kl_diff = kl_divergence - prev_kl_divergence
#                 adjustment = self.alpha * (-kl_diff).clamp(-0.1, 0.1)
#                 noise_std *= (1 + adjustment)

#             prev_kl_divergence = kl_divergence
            
#             kl_divergence_list.append(kl_divergence)
#             weight_map_list.append(predicted_noise)
        
#         min_kl_index = torch.argmin(torch.stack(kl_divergence_list))
#         best_weight_map = weight_map_list[min_kl_index]
        
#         weighted_average_weight_map = torch.sigmoid(best_weight_map)
#         return weighted_average_weight_map

# class Diffusion_PSA2(nn.Module):
#     def __init__(self, c1, c2, e=0.5, attn_ratio=0.5):
#         super().__init__()
#         assert(c1 == c2)
#         self.c = int(c1 * e)
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv(2 * self.c, c1, 1)
        
#         self.attn = Attention(self.c, attn_ratio=attn_ratio, num_heads=self.c // 64)
#         self.diffusion_attn = DiffusionAttention2(self.c, time_steps=10, alpha=0.1)
#         self.ffn = nn.Sequential(
#             Conv(self.c, self.c*2, 1),
#             Conv(self.c*2, self.c, 1, act=False)
#         )
        
#     def forward(self, x):
#         a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
#         diffusion_weight_map = self.diffusion_attn(b)
#         b = b * diffusion_weight_map
#         original_attn = self.attn(b)
#         b = b + original_attn
#         b = b + self.ffn(b)
        
#         return self.cv2(torch.cat((a, b), 1))


#基于kde和kl综合评估-------------version2

import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import numpy as np

class DiffusionAttention2(nn.Module):
    def __init__(self, channels, time_steps=10, alpha=0.1, epsilon=1e-5, kl_weight=0.5, pca_components=5, regularization=1e-5):
        super().__init__()
        self.channels = channels
        self.time_steps = time_steps
        self.alpha = alpha  # 平滑因子
        self.epsilon = epsilon  # 数值稳定性
        self.kl_weight = kl_weight  # KDE KL 散度的权重
        self.pca_components = pca_components  # PCA 降维后的组件数
        self.regularization = regularization  # 协方差矩阵正则化项
        self.denoise_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def traditional_kl_divergence(self, x, x_hat):
        mu_P = x.mean(dim=(0, 2, 3))
        mu_Q = x_hat.mean(dim=(0, 2, 3))
        sigma_P = x.var(dim=(0, 2, 3)) + self.epsilon
        sigma_Q = x_hat.var(dim=(0, 2, 3)) + self.epsilon

        kl_divergence = 0.5 * (torch.log(sigma_Q / sigma_P) + 
                               (sigma_P / sigma_Q) + 
                               (mu_P - mu_Q)**2 / sigma_Q - 
                               1).sum()
        return kl_divergence
    
    def kde_kl_divergence(self, x, x_hat):
        scaler = StandardScaler()
        x_np = scaler.fit_transform(x.flatten().detach().cpu().numpy().reshape(-1, 1))
        x_hat_np = scaler.transform(x_hat.flatten().detach().cpu().numpy().reshape(-1, 1))

        # 动态调整 n_components 确保其合法性
        pca_components = min(self.pca_components, x_np.shape[0], x_np.shape[1], x_hat_np.shape[0], x_hat_np.shape[1])
        pca = PCA(n_components=pca_components)

        try:
            x_pca = pca.fit_transform(x_np)
            x_hat_pca = pca.transform(x_hat_np)
        except ValueError as e:
            # 如果 PCA 失败，直接返回零 KL 散度
            return torch.tensor(0.0, device=x.device)

        # 计算总方差
        total_var = pca.explained_variance_.sum()

        # 如果 total_var 非常小，则跳过解释方差比率的计算
        if total_var < self.epsilon:
            return torch.tensor(0.0, device=x.device)

        explained_variance_ratio_ = pca.explained_variance_ / total_var  # 正常情况下不再加 epsilon

        if np.isnan(explained_variance_ratio_).any():
            return torch.tensor(0.0, device=x.device)
        
        if x_pca.shape[1] <= 1:
            return torch.tensor(0.0, device=x.device)

        kde_p = gaussian_kde(x_pca.T, bw_method='silverman')
        
        # 自适应的正则化，防止协方差矩阵接近奇异
        regularization_strength = max(self.regularization, np.linalg.norm(pca.explained_variance_))
        kde_p.covariance += np.eye(kde_p.covariance.shape[0]) * regularization_strength

        kde_q = gaussian_kde(x_hat_pca.T, bw_method='silverman')
        kde_q.covariance += np.eye(kde_q.covariance.shape[0]) * regularization_strength

        p_vals = kde_p(x_pca.T) + self.epsilon
        q_vals = kde_q(x_hat_pca.T) + self.epsilon
        
        kl_divergence = np.sum(p_vals * np.log(p_vals / q_vals))
        return torch.tensor(kl_divergence, device=x.device)
    
    def combined_kl_divergence(self, x, x_hat):
        traditional_kl = self.traditional_kl_divergence(x, x_hat)
        kde_kl = self.kde_kl_divergence(x, x_hat)
        
        combined_kl = (1 - self.kl_weight) * traditional_kl + self.kl_weight * kde_kl
        return combined_kl
        
    def forward(self, x):
        T = self.time_steps
        alphas = torch.linspace(0.9, 0.1, T).to(x.device)
        alpha_bars = torch.cumprod(alphas, dim=0).to(x.device)
        
        noise_std = 1.0
        noise = torch.randn_like(x)
        x_t = torch.sqrt(alpha_bars[-1] + self.epsilon) * x + torch.sqrt(1 - alpha_bars[-1] + self.epsilon) * noise
        weight_map_list = []
        kl_divergence_list = []

        prev_kl_divergence = None
        
        for t in range(T):
            predicted_noise = self.denoise_net(x_t)
            x_hat = (x_t - torch.sqrt(1 - alphas[t] + self.epsilon) * predicted_noise) / torch.sqrt(alpha_bars[t] + self.epsilon)
            
            kl_divergence = self.combined_kl_divergence(x, x_hat)

            if prev_kl_divergence is not None:
                kl_diff = kl_divergence - prev_kl_divergence
                adjustment = self.alpha * (-kl_diff).clamp(-0.1, 0.1)
                noise_std *= (1 + adjustment)

            prev_kl_divergence = kl_divergence
            
            kl_divergence_list.append(kl_divergence)
            weight_map_list.append(predicted_noise)
        
        min_kl_index = torch.argmin(torch.stack(kl_divergence_list))
        best_weight_map = weight_map_list[min_kl_index]
        
        weighted_average_weight_map = torch.sigmoid(best_weight_map)
        return weighted_average_weight_map

class Diffusion_PSA2(nn.Module):
    def __init__(self, c1, c2, e=0.5, attn_ratio=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=attn_ratio, num_heads=self.c // 64)
        self.diffusion_attn = DiffusionAttention2(self.c, time_steps=10, alpha=0.1)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
        diffusion_weight_map = self.diffusion_attn(b)
        b = b * diffusion_weight_map
        original_attn = self.attn(b)
        b = b + original_attn
        b = b + self.ffn(b)
        
        return self.cv2(torch.cat((a, b), 1))

###核pca版 psa2
import torch
import torch.nn as nn
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import numpy as np

class DiffusionAttention3(nn.Module):
    def __init__(self, channels, time_steps=10, alpha=0.1, epsilon=1e-5, kl_weight=0.5, pca_components=5, regularization=1e-5, kernel='lineat', gamma=None):
        super().__init__()
        self.channels = channels
        self.time_steps = time_steps
        self.alpha = alpha  # 平滑因子
        self.epsilon = epsilon  # 数值稳定性
        self.kl_weight = kl_weight  # KDE KL 散度的权重
        self.pca_components = pca_components  # 核PCA 降维后的组件数
        self.regularization = regularization  # 协方差矩阵正则化项
        self.kernel = kernel  # 核函数类型
        self.gamma = gamma  # 核函数的gamma参数
        self.denoise_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def traditional_kl_divergence(self, x, x_hat):
        mu_P = x.mean(dim=(0, 2, 3))
        mu_Q = x_hat.mean(dim=(0, 2, 3))
        sigma_P = x.var(dim=(0, 2, 3)) + self.epsilon
        sigma_Q = x_hat.var(dim=(0, 2, 3)) + self.epsilon

        kl_divergence = 0.5 * (torch.log(sigma_Q / sigma_P) + 
                               (sigma_P / sigma_Q) + 
                               (mu_P - mu_Q)**2 / sigma_Q - 
                               1).sum()
        return kl_divergence
    
    def kde_kl_divergence(self, x, x_hat):
        scaler = StandardScaler()
        x_np = scaler.fit_transform(x.flatten().detach().cpu().numpy().reshape(-1, 1))
        x_hat_np = scaler.transform(x_hat.flatten().detach().cpu().numpy().reshape(-1, 1))

        # 动态调整 n_components 确保其合法性
        pca_components = min(self.pca_components, x_np.shape[0], x_np.shape[1], x_hat_np.shape[0], x_hat_np.shape[1])
        kpca = KernelPCA(n_components=pca_components, kernel=self.kernel, gamma=self.gamma)

        try:
            x_pca = kpca.fit_transform(x_np)
            x_hat_pca = kpca.transform(x_hat_np)
        except ValueError as e:
            # 如果 KernelPCA 失败，直接返回零 KL 散度
            return torch.tensor(0.0, device=x.device)

        # 计算总方差（可以使用核PCA的解释方差）
        total_var = np.var(x_pca, axis=0).sum()

        # 如果 total_var 非常小，则跳过解释方差比率的计算
        if total_var < self.epsilon:
            return torch.tensor(0.0, device=x.device)

        explained_variance_ratio_ = np.var(x_pca, axis=0) / total_var

        if np.isnan(explained_variance_ratio_).any():
            return torch.tensor(0.0, device=x.device)
        
        if x_pca.shape[1] <= 1:
            return torch.tensor(0.0, device=x.device)

        kde_p = gaussian_kde(x_pca.T, bw_method='silverman')
        
        # 自适应的正则化，防止协方差矩阵接近奇异
        regularization_strength = max(self.regularization, np.linalg.norm(np.var(x_pca, axis=0)))
        kde_p.covariance += np.eye(kde_p.covariance.shape[0]) * regularization_strength

        kde_q = gaussian_kde(x_hat_pca.T, bw_method='silverman')
        kde_q.covariance += np.eye(kde_q.covariance.shape[0]) * regularization_strength

        p_vals = kde_p(x_pca.T) + self.epsilon
        q_vals = kde_q(x_hat_pca.T) + self.epsilon
        
        kl_divergence = np.sum(p_vals * np.log(p_vals / q_vals))
        return torch.tensor(kl_divergence, device=x.device)
    
    def combined_kl_divergence(self, x, x_hat):
        traditional_kl = self.traditional_kl_divergence(x, x_hat)
        kde_kl = self.kde_kl_divergence(x, x_hat)
        
        combined_kl = (1 - self.kl_weight) * traditional_kl + self.kl_weight * kde_kl
        return combined_kl
        
    def forward(self, x):
        T = self.time_steps
        alphas = torch.linspace(0.9, 0.1, T).to(x.device)
        alpha_bars = torch.cumprod(alphas, dim=0).to(x.device)
        
        noise_std = 1.0
        noise = torch.randn_like(x)
        x_t = torch.sqrt(alpha_bars[-1] + self.epsilon) * x + torch.sqrt(1 - alpha_bars[-1] + self.epsilon) * noise
        weight_map_list = []
        kl_divergence_list = []

        prev_kl_divergence = None
        
        for t in range(T):
            predicted_noise = self.denoise_net(x_t)
            x_hat = (x_t - torch.sqrt(1 - alphas[t] + self.epsilon) * predicted_noise) / torch.sqrt(alpha_bars[t] + self.epsilon)
            
            kl_divergence = self.combined_kl_divergence(x, x_hat)

            if prev_kl_divergence is not None:
                kl_diff = kl_divergence - prev_kl_divergence
                adjustment = self.alpha * (-kl_diff).clamp(-0.1, 0.1)
                noise_std *= (1 + adjustment)

            prev_kl_divergence = kl_divergence
            
            kl_divergence_list.append(kl_divergence)
            weight_map_list.append(predicted_noise)
        
        min_kl_index = torch.argmin(torch.stack(kl_divergence_list))
        best_weight_map = weight_map_list[min_kl_index]
        
        weighted_average_weight_map = torch.sigmoid(best_weight_map)
        return weighted_average_weight_map

class Diffusion_PSA3(nn.Module):
    def __init__(self, c1, c2, e=0.5, attn_ratio=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=attn_ratio, num_heads=self.c // 64)
        self.diffusion_attn = DiffusionAttention3(self.c, time_steps=10, alpha=0.1)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
        diffusion_weight_map = self.diffusion_attn(b)
        b = b * diffusion_weight_map
        original_attn = self.attn(b)
        b = b + original_attn
        b = b + self.ffn(b)
        
        return self.cv2(torch.cat((a, b), 1))



####基于vi的diffpsa

import torch
import torch.nn as nn
import torch.distributions as dist

class DiffusionAttention4(nn.Module):
    def __init__(self, channels, time_steps=10, alpha=0.1, epsilon=1e-5, kl_weight=0.5):
        super().__init__()
        self.channels = channels
        self.time_steps = time_steps
        self.alpha = alpha  # 平滑因子
        self.epsilon = epsilon  # 数值稳定性
        self.kl_weight = kl_weight  # VI KL 散度的权重
        self.denoise_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        
    def traditional_kl_divergence(self, x, x_hat):
        mu_P = x.mean(dim=(0, 2, 3))
        mu_Q = x_hat.mean(dim=(0, 2, 3))
        sigma_P = x.var(dim=(0, 2, 3)) + self.epsilon
        sigma_Q = x_hat.var(dim=(0, 2, 3)) + self.epsilon

        kl_divergence = 0.5 * (torch.log(sigma_Q / sigma_P) + 
                               (sigma_P / sigma_Q) + 
                               (mu_P - mu_Q)**2 / sigma_Q - 
                               1).sum()
        return kl_divergence
    
    def vi_kl_divergence(self, x, x_hat):
        # 假设潜在表示为正态分布
        mu_P = x.mean(dim=(0, 2, 3))
        mu_Q = x_hat.mean(dim=(0, 2, 3))
        sigma_P = x.var(dim=(0, 2, 3)).sqrt() + self.epsilon
        sigma_Q = x_hat.var(dim=(0, 2, 3)).sqrt() + self.epsilon

        # 定义高斯分布
        p = dist.Normal(mu_P, sigma_P)
        q = dist.Normal(mu_Q, sigma_Q)
        
        # 计算 KL 散度
        kl_divergence = dist.kl_divergence(p, q).sum()
        return kl_divergence
    
    def combined_kl_divergence(self, x, x_hat):
        traditional_kl = self.traditional_kl_divergence(x, x_hat)
        vi_kl = self.vi_kl_divergence(x, x_hat)
        
        # 加权组合
        combined_kl = (1 - self.kl_weight) * traditional_kl + self.kl_weight * vi_kl
        return combined_kl
        
    def forward(self, x):
        T = self.time_steps
        alphas = torch.linspace(0.9, 0.1, T).to(x.device)
        alpha_bars = torch.cumprod(alphas, dim=0).to(x.device)
        
        noise_std = 1.0  # 初始噪声强度
        noise = torch.randn_like(x)
        x_t = torch.sqrt(alpha_bars[-1] + self.epsilon) * x + torch.sqrt(1 - alpha_bars[-1] + self.epsilon) * noise
        weight_map_list = []
        kl_divergence_list = []
        
        prev_kl_divergence = None
        
        for t in range(T):
            predicted_noise = self.denoise_net(x_t)
            x_hat = (x_t - torch.sqrt(1 - alphas[t] + self.epsilon) * predicted_noise) / torch.sqrt(alpha_bars[t] + self.epsilon)
            
            # 使用加权组合的 KL 散度
            kl_divergence = self.combined_kl_divergence(x, x_hat)
            
            # 调整噪声强度
            if prev_kl_divergence is not None:
                kl_diff = kl_divergence - prev_kl_divergence
                adjustment = self.alpha * (-kl_diff).clamp(-0.1, 0.1)  # 使用平滑的调整策略
                noise_std *= (1 + adjustment)
            
            prev_kl_divergence = kl_divergence
            
            kl_divergence_list.append(kl_divergence)
            weight_map_list.append(predicted_noise)
        
        # 选择与原始特征图散度最小的时间步所对应的权重图
        min_kl_index = torch.argmin(torch.stack(kl_divergence_list))
        best_weight_map = weight_map_list[min_kl_index]
        
        weighted_average_weight_map = torch.sigmoid(best_weight_map)
        return weighted_average_weight_map

class Diffusion_PSA4(nn.Module):
    def __init__(self, c1, c2, e=0.5, attn_ratio=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=attn_ratio, num_heads=self.c // 64)
        self.diffusion_attn = DiffusionAttention4(self.c, time_steps=10, alpha=0.1)  # 传入平滑因子
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
        # 生成扩散过程的权重图
        diffusion_weight_map = self.diffusion_attn(b)
    
        # 在自注意力机制之前应用权重图
        b = b * diffusion_weight_map
    
        # 原始自注意力机制
        original_attn = self.attn(b)
    
        b = b + original_attn
        b = b + self.ffn(b)
        
        return self.cv2(torch.cat((a, b), 1))



######svd diffpsa

import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
import numpy as np

class DiffusionAttention6(nn.Module):
    def __init__(self, channels, time_steps=10, alpha=0.1, epsilon=1e-5, kl_weight=0.5, pca_components=5, regularization=1e-5):
        super().__init__()
        self.channels = channels
        self.time_steps = time_steps
        self.alpha = alpha  # 平滑因子
        self.epsilon = epsilon  # 数值稳定性
        self.kl_weight = kl_weight  # KDE KL 散度的权重
        self.pca_components = pca_components  # PCA 降维后的组件数
        self.regularization = regularization  # 协方差矩阵正则化项
        self.denoise_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def traditional_kl_divergence(self, x, x_hat):
        mu_P = x.mean(dim=(0, 2, 3))
        mu_Q = x_hat.mean(dim=(0, 2, 3))
        sigma_P = x.var(dim=(0, 2, 3)) + self.epsilon
        sigma_Q = x_hat.var(dim=(0, 2, 3)) + self.epsilon

        kl_divergence = 0.5 * (torch.log(sigma_Q / sigma_P) + 
                               (sigma_P / sigma_Q) + 
                               (mu_P - mu_Q)**2 / sigma_Q - 
                               1).sum()
        return kl_divergence
    
    def kde_kl_divergence(self, x, x_hat):
        scaler = StandardScaler()
        x_np = scaler.fit_transform(x.flatten().detach().cpu().numpy().reshape(-1, 1))
        x_hat_np = scaler.transform(x_hat.flatten().detach().cpu().numpy().reshape(-1, 1))

        # 动态调整 n_components 确保其合法性
        pca_components = min(self.pca_components, x_np.shape[0], x_np.shape[1], x_hat_np.shape[0], x_hat_np.shape[1])
        svd = TruncatedSVD(n_components=pca_components)

        try:
            x_pca = svd.fit_transform(x_np)
            x_hat_pca = svd.transform(x_hat_np)
        except ValueError as e:
            # 如果 SVD 失败，返回一个小的非零 KL 散度
            return torch.tensor(1e-6, device=x.device)

        # 计算总方差
        total_var = np.var(x_pca, axis=0).sum()

        # 如果 total_var 非常小，则跳过解释方差比率的计算
        if total_var < self.epsilon:
            return torch.tensor(0.0, device=x.device)

        explained_variance_ratio_ = np.var(x_pca, axis=0) / total_var

        if np.isnan(explained_variance_ratio_).any():
            return torch.tensor(0.0, device=x.device)
        
        if x_pca.shape[1] <= 1:
            return torch.tensor(0.0, device=x.device)

        kde_p = gaussian_kde(x_pca.T, bw_method='silverman')
        
        # 自适应的正则化，防止协方差矩阵接近奇异
        regularization_strength = max(self.regularization, np.linalg.norm(np.var(x_pca, axis=0)))
        kde_p.covariance += np.eye(kde_p.covariance.shape[0]) * regularization_strength

        kde_q = gaussian_kde(x_hat_pca.T, bw_method='silverman')
        kde_q.covariance += np.eye(kde_q.covariance.shape[0]) * regularization_strength

        p_vals = kde_p(x_pca.T) + self.epsilon
        q_vals = kde_q(x_hat_pca.T) + self.epsilon
        
        kl_divergence = np.sum(p_vals * np.log(p_vals / q_vals))
        return torch.tensor(kl_divergence, device=x.device)
    
    def combined_kl_divergence(self, x, x_hat):
        traditional_kl = self.traditional_kl_divergence(x, x_hat)
        kde_kl = self.kde_kl_divergence(x, x_hat)
        
        combined_kl = (1 - self.kl_weight) * traditional_kl + self.kl_weight * kde_kl
        return combined_kl
        
    def forward(self, x):
        T = self.time_steps
        alphas = torch.linspace(0.9, 0.1, T).to(x.device)
        alpha_bars = torch.cumprod(alphas, dim=0).to(x.device)
        
        noise_std = 1.0
        noise = torch.randn_like(x)
        x_t = torch.sqrt(alpha_bars[-1] + self.epsilon) * x + torch.sqrt(1 - alpha_bars[-1] + self.epsilon) * noise
        weight_map_list = []
        kl_divergence_list = []

        prev_kl_divergence = None
        
        for t in range(T):
            predicted_noise = self.denoise_net(x_t)
            x_hat = (x_t - torch.sqrt(1 - alphas[t] + self.epsilon) * predicted_noise) / torch.sqrt(alpha_bars[t] + self.epsilon)
            
            kl_divergence = self.combined_kl_divergence(x, x_hat)

            if prev_kl_divergence is not None:
                kl_diff = kl_divergence - prev_kl_divergence
                adjustment = self.alpha * (-kl_diff).clamp(-0.1, 0.1)
                noise_std *= (1 + adjustment)

            prev_kl_divergence = kl_divergence
            
            kl_divergence_list.append(kl_divergence)
            weight_map_list.append(predicted_noise)
        
        min_kl_index = torch.argmin(torch.stack(kl_divergence_list))
        best_weight_map = weight_map_list[min_kl_index]
        
        weighted_average_weight_map = torch.sigmoid(best_weight_map)
        return weighted_average_weight_map

class Diffusion_PSA6(nn.Module):
    def __init__(self, c1, c2, e=0.5, attn_ratio=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=attn_ratio, num_heads=self.c // 64)
        self.diffusion_attn = DiffusionAttention6(self.c, time_steps=10, alpha=0.1)
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
        diffusion_weight_map = self.diffusion_attn(b)
        b = b * diffusion_weight_map
        original_attn = self.attn(b)
        b = b + original_attn
        b = b + self.ffn(b)
        
        return self.cv2(torch.cat((a, b), 1))



######输出特征图形状

import torch
import torch.nn as nn

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

class DiffusionAttention7(nn.Module):
    def __init__(self, channels, time_steps=10, alpha=0.1, epsilon=1e-5, save_path='/home/wuz/fhb_project/output_log/map-out'):
        super().__init__()
        self.channels = channels
        self.time_steps = time_steps
        self.alpha = alpha  # 平滑因子
        self.epsilon = epsilon  # 数值稳定性
        self.denoise_net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
    def save_feature_map(self, feature_map, step, stage):
        batch_size, channels, height, width = feature_map.shape
        feature_map_np = feature_map[0].cpu().detach().numpy()  # 假设 batch size >= 1, 取出第一个样本
        plt.figure(figsize=(10, 10))
        for i in range(min(16, channels)):  # 取前16个通道进行可视化
            plt.subplot(4, 4, i+1)
            plt.imshow(feature_map_np[i], cmap='gray')
            plt.axis('off')
        plt.suptitle(f"Stage: {stage}, Time Step: {step}")
        plt.savefig(os.path.join(self.save_path, f"{stage}_timestep_{step}.png"))
        plt.close()
        
    def forward(self, x):
        print(f"Input feature map shape: {x.shape}")
        self.save_feature_map(x, step=0, stage="input")
        
        T = self.time_steps
        alphas = torch.linspace(0.9, 0.1, T).to(x.device)
        alpha_bars = torch.cumprod(alphas, dim=0).to(x.device)
        
        noise_std = 1.0  # 初始噪声强度
        noise = torch.randn_like(x)
        x_t = torch.sqrt(alpha_bars[-1] + self.epsilon) * x + torch.sqrt(1 - alpha_bars[-1] + self.epsilon) * noise
        weight_map_list = []
        kl_divergence_list = []
        
        prev_kl_divergence = None
        
        for t in range(T):
            predicted_noise = self.denoise_net(x_t)
            print(f"Predicted denoised feature map shape at time step {t}: {predicted_noise.shape}")
            self.save_feature_map(predicted_noise, step=t, stage="predicted_denoised")
            
            x_hat = (x_t - torch.sqrt(1 - alphas[t] + self.epsilon) * predicted_noise) / torch.sqrt(alpha_bars[t] + self.epsilon)
            
            # 使用均值和方差计算 KL 散度，添加 epsilon 以防止数值问题
            mu_P = x.mean(dim=(0, 2, 3))
            mu_Q = x_hat.mean(dim=(0, 2, 3))
            sigma_P = x.var(dim=(0, 2, 3)) + self.epsilon
            sigma_Q = x_hat.var(dim=(0, 2, 3)) + self.epsilon

            kl_divergence = 0.5 * (torch.log(sigma_Q / sigma_P) + 
                                   (sigma_P / sigma_Q) + 
                                   (mu_P - mu_Q)**2 / sigma_Q - 
                                   1).sum()
            
            # 调整噪声强度
            if prev_kl_divergence is not None:
                kl_diff = kl_divergence - prev_kl_divergence
                adjustment = self.alpha * (-kl_diff).clamp(-0.1, 0.1)  # 使用平滑的调整策略
                noise_std *= (1 + adjustment)
            
            prev_kl_divergence = kl_divergence
            
            kl_divergence_list.append(kl_divergence)
            weight_map_list.append(predicted_noise)
        
        # 选择与原始特征图散度最小的时间步所对应的权重图
        min_kl_index = torch.argmin(torch.stack(kl_divergence_list))
        best_weight_map = weight_map_list[min_kl_index]
        
        print(f"Output weighted map shape: {best_weight_map.shape}")
        self.save_feature_map(best_weight_map, step=min_kl_index.item(), stage="output_weighted_map")
        
        weighted_average_weight_map = torch.sigmoid(best_weight_map)
        return weighted_average_weight_map

# 使用示例：
# diffusion_attn = DiffusionAttention7(channels=128, time_steps=10, alpha=0.1, save_path="your_save_directory")
# 注意：需要替换 `your_save_directory` 为你想要保存的目录路径


class Diffusion_PSA7(nn.Module):
    def __init__(self, c1, c2, e=0.5, attn_ratio=0.5):
        super().__init__()
        assert(c1 == c2)
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=attn_ratio, num_heads=self.c // 64)
        self.diffusion_attn = DiffusionAttention7(self.c, time_steps=10, alpha=0.1)  # 传入平滑因子
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
        # 生成扩散过程的权重图
        diffusion_weight_map = self.diffusion_attn(b)
    
        # 在自注意力机制之前应用权重图
        b = b * diffusion_weight_map
    
        # 原始自注意力机制
        original_attn = self.attn(b)
    
        b = b + original_attn
        b = b + self.ffn(b)
        
        return self.cv2(torch.cat((a, b), 1))



import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F
from geomloss import SamplesLoss  # 用于计算Wasserstein距离
from models.radon_module import Radon  # 导入您提供的Radon实现
from pytorch_wavelets import DWTForward  # 从pytorch_wavelets库导入DWTForward

class SharedEncoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(SharedEncoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, latent_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.squeeze(-1).squeeze(-1)  # Flatten to (batch_size, latent_dim)

class SharedDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels, original_size):
        super(SharedDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 128 * original_size[0] * original_size[1])
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 64, kernel_size=3, padding=1)
        self.deconv1 = nn.ConvTranspose2d(64, output_channels, kernel_size=3, padding=1)

    def forward(self, z, original_size):
        z = self.fc(z)
        z = z.view(-1, 128, original_size[0], original_size[1])
        z = F.relu(self.deconv3(z))
        z = F.relu(self.deconv2(z))
        z = torch.sigmoid(self.deconv1(z))
        return z

class CrossDomainFeatureSelection(nn.Module):
    def __init__(self, channels, latent_dim=128, epsilon=1e-5, sparsity_type='euclidean'):
        super().__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.epsilon = epsilon
        self.sparsity_type = sparsity_type

        # Shared encoder and decoder
        self.shared_encoder = SharedEncoder(channels, latent_dim)
        
        # 1x1卷积层用于恢复通道数
        self.conv1x1 = nn.Conv2d(3 * channels, channels, kernel_size=1)
        
        # Wasserstein distance loss function
        self.transport_loss = SamplesLoss("sinkhorn", p=2, blur=0.05)
        
        # Initialize Radon transform
        self.radon_transform = Radon(in_size=None, theta=torch.arange(180))  # Radon初始化

        # Initialize DWTForward for wavelet transform
        self.dwt = DWTForward(J=1, wave='haar', mode='zero')  # 默认1级小波变换

    def compute_sparsity(self, x):
        if self.sparsity_type == 'euclidean':
            sparsity = torch.sqrt(torch.sum(x ** 2, dim=(2, 3), keepdim=True))
        elif self.sparsity_type == 'manhattan':
            sparsity = torch.sum(torch.abs(x), dim=(2, 3), keepdim=True)
        else:
            raise ValueError("Unsupported sparsity type. Use 'euclidean' or 'manhattan'.")
        return sparsity

    def forward(self, b):
        device = b.device  # 确保所有张量在同一设备上
        original_size = b.size()[2:]  # 获取输入的原始尺寸
        
        # Clone the input for different transformations
        b1, b2, b3 = b, b, b
        
        # Frequency Domain (Fourier Transform) on b1
        freq_b1 = torch.fft.fft2(b1)
        freq_b1 = torch.fft.fftshift(freq_b1)  # Center the zero frequency component
        freq_b1 = freq_b1.abs().to(device)
        freq_sparsity = self.compute_sparsity(freq_b1).to(device)

        # Wavelet Domain (Full Wavelet Transform) on b2
        yl, yh = self.dwt(b2)
        wavelet_b2 = yl.to(device)  # 只使用低频分量
        wavelet_sparsity = self.compute_sparsity(wavelet_b2).to(device)

        # Radon Transform (Full Radon Transform) on b3 using the provided Radon class
        radon_b3 = self.radon_transform(b3.to(device)).to(device)
        radon_sparsity = self.compute_sparsity(radon_b3).to(device)

        # Encode all three domains into a common feature space
        encoded_freq = self.shared_encoder(freq_b1).to(device)
        encoded_wavelet = self.shared_encoder(wavelet_b2).to(device)
        encoded_radon = self.shared_encoder(radon_b3).to(device)

        # Compute optimal transport weights between encoded features
        ot_loss_fw = self.transport_loss(encoded_freq.view(freq_b1.size(0), -1).to(device),
                                         encoded_wavelet.view(wavelet_b2.size(0), -1).to(device))
        ot_loss_fr = self.transport_loss(encoded_freq.view(freq_b1.size(0), -1).to(device),
                                         encoded_radon.view(radon_b3.size(0), -1).to(device))

        # Combine transport weights
        transport_weight_fw = torch.exp(-ot_loss_fw).to(device)
        transport_weight_fr = torch.exp(-ot_loss_fr).to(device)
        total_weight = transport_weight_fw.sum() + transport_weight_fr.sum()

        freq_weight = (transport_weight_fw / total_weight).to(device)
        wavelet_weight = (transport_weight_fw / total_weight).to(device)
        radon_weight = (transport_weight_fr / total_weight).to(device)

        # Decode weights back to original feature space
        decoder_freq = SharedDecoder(self.latent_dim, self.channels, original_size).to(device)
        decoder_wavelet = SharedDecoder(self.latent_dim, self.channels, original_size).to(device)
        decoder_radon = SharedDecoder(self.latent_dim, self.channels, original_size).to(device)

        # Apply weights to the decoded features
        decoded_freq_weight = decoder_freq(encoded_freq * freq_weight, original_size).to(device)
        decoded_wavelet_weight = decoder_wavelet(encoded_wavelet * wavelet_weight, original_size).to(device)
        decoded_radon_weight = decoder_radon(encoded_radon * radon_weight, original_size).to(device)

        # Apply weights to original domains
        b1_weighted = b1 * decoded_freq_weight * freq_sparsity
        b2_weighted = b2 * decoded_wavelet_weight * wavelet_sparsity
        b3_weighted = b3 * decoded_radon_weight * radon_sparsity

        # Concatenate along the channel dimension
        combined_b = torch.cat((b1_weighted, b2_weighted, b3_weighted), dim=1).to(device)

        # Use 1x1 convolution to reduce the channel dimension back to the original
        combined_b = self.conv1x1(combined_b).to(device)

        # Apply sigmoid to get the final weight map
        final_weight_map = torch.sigmoid(combined_b)

        return final_weight_map
    

class CrossDomain_PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5, attn_ratio=0.5, sparsity_type='euclidean'):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        
        self.attn = Attention(self.c, attn_ratio=attn_ratio, num_heads=self.c // 64)
        self.cross_domain_feat_sel = CrossDomainFeatureSelection(self.c, sparsity_type=sparsity_type)  # 使用跨域特征选择模块
        self.ffn = nn.Sequential(
            Conv(self.c, self.c*2, 1),
            Conv(self.c*2, self.c, 1, act=False)
        )
        
    def forward(self, x):
        device = x.device  # 确保所有张量在同一设备上
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        
        # 生成跨域特征选择的权重图
        cross_domain_weight_map = self.cross_domain_feat_sel(b)
    
        # 在自注意力机制之前应用权重图
        b = b * cross_domain_weight_map
    
        # 原始自注意力机制
        original_attn = self.attn(b)
    
        b = b + original_attn
        b = b + self.ffn(b)
        
        return self.cv2(torch.cat((a, b), 1))
    
######cd-gelan4
import torch
import torch.nn as nn
import torch.fft
from torch.cuda.amp import autocast
from pytorch_wavelets import DWTForward, DWTInverse  # 用于小波变换

class WaveletAttentionModule(nn.Module):
    def __init__(self, wave='haar'):
        super(WaveletAttentionModule, self).__init__()
        # Initialize DWT and IDWT
        self.dwt = DWTForward(J=1, wave=wave)
        self.idwt = DWTInverse(wave=wave)
        self.attention_low = None
        self.attention_high = None
        self.last_C = None

    def forward(self, x):
        # Disable autocast for the wavelet operations
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()  # Ensure x is float32
            B, C, H, W = x.shape
            print(f"WaveletAttentionModule input x shape: {x.shape}")  # Debug

            # Perform DWT
            yl, yh = self.dwt(x)
            yh = yh[0]  # [B, C, D, H', W']
            print(f"yl shape: {yl.shape}, yh shape: {yh.shape}")  # Debug

            # Initialize attention_low if needed
            if self.attention_low is None or self.last_C != C:
                self.attention_low = nn.Sequential(
                    nn.Conv2d(C, C, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(C, C, kernel_size=1),
                    nn.Sigmoid()
                ).to(x.device)
                self.add_module('attention_low', self.attention_low)
                self.last_C = C

            # Apply attention to yl
            attn_yl = self.attention_low(yl)
            yl_enhanced = yl * attn_yl

            # Reshape yh correctly
            yh = yh.permute(0, 1, 3, 4, 2)  # [B, C, H', W', D]
            yh = yh.reshape(B, -1, yh.shape[2], yh.shape[3])  # [B, C*D, H', W']
            print(f"yh shape after reshape: {yh.shape}")  # Debug

            # Initialize attention_high if needed
            in_channels = yh.size(1)  # Should be C * D
            if self.attention_high is None or self.last_C != C:
                self.attention_high = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels, in_channels, kernel_size=1),
                    nn.Sigmoid()
                ).to(x.device)
                self.add_module('attention_high', self.attention_high)

            # Apply attention to yh
            attn_yh = self.attention_high(yh)
            yh_enhanced = yh * attn_yh

            # Reshape yh_enhanced back to [B, C, D, H', W']
            yh_enhanced = yh_enhanced.view(B, C, -1, yh.shape[2], yh.shape[3])  # [B, C, D, H', W']
            yh_enhanced = yh_enhanced.permute(0, 1, 2, 3, 4)  # [B, C, D, H', W']
            yh_enhanced = [yh_enhanced]  # Wrap in a list as expected by IDWT
            print(f"yh_enhanced shape after reshape back: {yh_enhanced[0].shape}")  # Debug

            # Perform IDWT
            x_reconstructed = self.idwt((yl_enhanced, yh_enhanced))  # [B, C, H, W]
            return x_reconstructed



class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 全局最大池化

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2  # 确保输出尺寸不变
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道维度求平均
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道维度求最大值
        x = torch.cat([avg_out, max_out], dim=1)  # 拼接在通道维度
        x = self.conv(x)
        return self.sigmoid(x)

class SA_GELAN_FFT4(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super(SA_GELAN_FFT4, self).__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Conv2d(c1, 2 * self.c, kernel_size=1, stride=1)
        # 添加降维卷积层
        self.reduce_conv = nn.Conv2d(3 * self.c, self.c, kernel_size=1)
        # 两个 PSA 块
        self.block1 = PSA(self.c, self.c, e)
        self.block2 = PSA(self.c, self.c, e)
        # 调整 cv2 的输入通道数
        self.cv2 = nn.Conv2d(3 * self.c, c1, kernel_size=1, stride=1)
        self.residual_conv = nn.Conv2d(c1, c1, kernel_size=1, stride=1)

        # 频域注意力模块
        self.fft_attention = MultiScaleFFTAttentionModule(self.c)
        # 小波域注意力模块
        self.wavelet_attention = WaveletAttentionModule()
        # 空域注意力模块（包含通道和空间注意力）
        self.channel_attention = ChannelAttention(self.c)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        with torch.cuda.amp.autocast():
        # Initial split into a and b
            a, b = self.cv1(x).split((self.c, self.c), dim=1)

        # Disable autocast for wavelet attention
            with torch.cuda.amp.autocast(enabled=False):
                b_wavelet = self.wavelet_attention(b.float())  # Cast b to float32

        # Continue with autocast enabled
        # 频域注意力增强
            b_fft = self.fft_attention(b)
        # 空域注意力增强
            b_ca = self.channel_attention(b) * b
            b_sa = self.spatial_attention(b) * b
            b_spatial = b_ca + b_sa

        # Merge features
            b_combined = torch.cat([b_fft, b_wavelet.type_as(b_fft), b_spatial], dim=1)  # 通道数为 3 * self.c

            # 使用 reduce_conv 降维
            b_reduced = self.reduce_conv(b_combined)  # 通道数变为 self.c

            # 通过两个 PSA 块
            m = self.block1(b_reduced)
            n = self.block2(m)

            # 拼接并经过最终的卷积
            combined = torch.cat([a, m, n], dim=1)  # 通道数为 self.c + self.c + self.c = 3 * self.c
            output = self.cv2(combined)

            # 添加残差连接
            output += self.residual_conv(x)
        return output
