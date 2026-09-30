import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==============================================================================
# 1. 基础生成器 (Backbone Generator) - 保持不变
# ==============================================================================
class BackboneGenerator(nn.Module):
    def __init__(self, input_dim=256, ngf=64, img_size=32):
        super(BackboneGenerator, self).__init__()
        self.init_size = img_size // 4
        self.ngf = ngf
        
        self.l1 = nn.Sequential(
            nn.Linear(input_dim, ngf * 2 * self.init_size ** 2),
            nn.BatchNorm1d(ngf * 2 * self.init_size ** 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(ngf * 2),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ngf * 2, ngf, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, z_c):
        out = self.l1(z_c)
        out = out.view(out.shape[0], self.ngf * 2, self.init_size, self.init_size)
        img_feat = self.conv_blocks(out)
        return img_feat # [B, 64, 32, 32]

# ==============================================================================
# 2. 融合交互适配器 (Fusion Interaction Adapter) - [维度修复版]
#    手动处理维度转置，不使用 batch_first=True
# ==============================================================================
class FusionInteractionAdapter(nn.Module):
    def __init__(self, in_channels=64, z_s_dim=256, num_heads=4, num_tokens=8):
        super(FusionInteractionAdapter, self).__init__()
        self.in_channels = in_channels
        self.num_tokens = num_tokens
        
        # --- A. 纹理令牌化 (Texture Tokenizer) ---
        self.style_tokenizer = nn.Sequential(
            nn.Linear(z_s_dim, in_channels * num_tokens),
            nn.LayerNorm(in_channels * num_tokens),
            nn.LeakyReLU(0.2)
        )
        
        # --- B. 结构投影 (Structure Projection) ---
        self.query_proj = nn.Conv2d(in_channels, in_channels, 1)
        
        # --- C. 交叉注意力 (Cross-Attention) ---
        # 默认 batch_first=False，输入需为 [L, B, E]
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(in_channels)
        
        # --- D. FFN ---
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.GELU(),
            nn.Linear(in_channels * 4, in_channels)
        )
        self.norm2 = nn.LayerNorm(in_channels)
        
        # 最后的空间映射
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, base_feat, z_s):
        B, C, H, W = base_feat.shape
        
        # 1. 准备 Query (Visual Tokens)
        # 原始: [B, C, H, W] -> [B, H*W, C]
        visual_tokens = base_feat.flatten(2).permute(0, 2, 1)
        
        # 2. 准备 Key, Value (Style Tokens)
        # 原始: [B, z_dim] -> [B, K, C]
        style_tokens = self.style_tokenizer(z_s).view(B, self.num_tokens, C)
        
        # 3. [关键步骤] 手动维度转置: [Batch, Length, Dim] -> [Length, Batch, Dim]
        # 这是为了满足 MultiheadAttention 的默认输入要求
        q = self.norm1(visual_tokens).permute(1, 0, 2) # [H*W, B, C]
        k = style_tokens.permute(1, 0, 2)              # [K, B, C]
        v = style_tokens.permute(1, 0, 2)              # [K, B, C]
        
        # 4. Attention 计算
        # 输出 attn_out: [H*W, B, C]
        attn_out, _ = self.cross_attn(query=q, key=k, value=v)
        
        # 5. [关键步骤] 转置回来: [Length, Batch, Dim] -> [Batch, Length, Dim]
        attn_out = attn_out.permute(1, 0, 2) # [B, H*W, C]
        
        # Residual Connection 1
        x = visual_tokens + attn_out
        
        # 6. FFN Block
        # FFN 是逐位置操作的 (Point-wise)，所以 [B, L, C] 格式可以直接输入
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        # 7. 还原空间结构
        # [B, H*W, C] -> [B, C, H, W]
        fused_feat = x.permute(0, 2, 1).view(B, C, H, W)
        
        # 8. 最终平滑
        out = self.out_conv(fused_feat)
        
        return out

# ==============================================================================
# 3. 解码器 (Decoder) - 保持不变
# ==============================================================================
class FeatureDecoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=3):
        super(FeatureDecoder, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.Tanh()
        )
    def forward(self, x):
        return self.layers(x)

# ==============================================================================
# 4. 整合: 融合编码器模拟生成器
# ==============================================================================
class HierarchicalGenerator(nn.Module):
    def __init__(self, args, img_sizes):
        super(HierarchicalGenerator, self).__init__()
        
        self.nz = 256 
        self.z_s_dim = 256 
        self.ngf = 64
        
        # 1. 骨干
        self.gan1_backbone = BackboneGenerator(input_dim=self.nz, ngf=self.ngf, img_size=img_sizes)
        
        # 2. 融合交互适配器
        self.gan2_adapter = FusionInteractionAdapter(in_channels=self.ngf, z_s_dim=256, num_heads=4, num_tokens=16)
        
        # 3. 解码器
        self.decoder = FeatureDecoder(in_channels=self.ngf)
        
        self.args = args
        self.img_sizes = img_sizes

    def forward(self, z_c, z_s):
        # 结构
        base_feat = self.gan1_backbone(z_c) 
        
        # 融合
        fused_feat = self.gan2_adapter(base_feat, z_s) 
        
        # 解码
        final_image = self.decoder(fused_feat)
        
        return final_image, base_feat, fused_feat

    def reinit(self):
        return get_generator(self.args, self.img_sizes)

# ==============================================================================
# 6. 对外接口
# ==============================================================================
def get_generator(args, img_sizes):
    model = HierarchicalGenerator(args, img_sizes)
    return model.cuda()