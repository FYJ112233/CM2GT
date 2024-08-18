import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch.nn import functional as F
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x,  **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)





### ViT
class Co_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        x_vis, x_nir = torch.chunk(x, 2, dim=0)

        ### MHATT for VIS modality obtain q,k.
        qkv_vis = self.to_qkv(x_vis).chunk(3, dim=-1)
        q_vis, k_vis, v_vis = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_vis)

        ### MHATT for NiR modality obtain v.
        qkv_nir = self.to_qkv(x_nir).chunk(3, dim=-1)
        q_nir, k_nir, v_nir = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv_nir)

        ### ViS çš? q,k,v.
        dots_vis = torch.matmul(q_vis, k_vis.transpose(-1, -2)) * self.scale
        attn_vis = self.attend(dots_vis)
        out_vis = torch.matmul(attn_vis, v_vis)
        out_re_vis = rearrange(out_vis, 'b h n d -> b n (h d)')
        out_to_vis_single = self.to_out(out_re_vis)

        ### NiR çš„q,k,v
        dots_nir = torch.matmul(q_nir, k_nir.transpose(-1, -2)) * self.scale
        attn_nir = self.attend(dots_nir)
        out_nir = torch.matmul(attn_nir, v_nir)
        out_re_nir = rearrange(out_nir, 'b h n d -> b n (h d)')
        out_to_nir_single = self.to_out(out_re_nir)

        ### ViSçš? k, v; + NiR çš„q
        dots_vis = torch.matmul(q_nir, k_nir.transpose(-1, -2)) * self.scale
        attn_vis = self.attend(dots_vis)
        out_vis = torch.matmul(attn_vis, v_vis)
        out_re_vis = rearrange(out_vis, 'b h n d -> b n (h d)')
        out_to_vis_co = self.to_out(out_re_vis)

        ### NiRçš? k, v ; + ViS çš? q.
        dots_nir = torch.matmul(q_vis, k_nir.transpose(-1, -2)) * self.scale
        attn_nir = self.attend(dots_nir)
        out_nir = torch.matmul(attn_nir, v_nir)
        out_re_nir = rearrange(out_nir, 'b h n d -> b n (h d)')
        out_to_nir_co = self.to_out(out_re_nir)


        ### ç‰¹å¾èžåˆ
        out_vis_final = torch.mul(out_to_vis_single, out_to_vis_co)
        out_nir_final = torch.mul(out_to_nir_single, out_to_nir_co)

        out_to = torch.cat((out_vis_final, out_nir_final), 0)

        return out_to




class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        # x_vis, x_nir = torch.chunk(x, 2, dim=0)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out_re = rearrange(out, 'b h n d -> b n (h d)')
        out_to = self.to_out(out_re)
        return out_to



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, SpatialGoupEnhance(groups=55)),
                # PreNorm(dim, CoTAttention(55, kernel_size=3, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        # for att_CoT, attn, ff in self.layers:
        #     x_COT = att_CoT(x)
        #     x_att = attn(x)
        #     x = x_COT + x_att + x
        #     x_ff = ff(x)
        #     x =  x_ff + x

        ### baseline
        for attn, ff in self.layers:
            x_att = attn(x) # [32, 163, 2048]
            x = x_att + x # [32, 163, 2048]
            x_ff = ff(x) # [32, 163, 2048]
            x = x_ff + x # [32, 163, 2048]
        return x

class Co_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # PreNorm(dim, SpatialGoupEnhance(groups=55)),
                # PreNorm(dim, CoTAttention(55, kernel_size=3, heads = heads, dim_head = dim_head, dropout = dropout)), #### [1,3]->54+1=55, [1, 2]->162+1=163
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                # PreNorm(dim, Co_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, modal):
        ### Group + Cot + self + att-co
        for att_group, att_cot, att_self, attn_co, ff in self.layers:
            b, w, h = x.shape

            x_g = att_group(x)

            x_COT = att_cot(x)
            x_COT_g = att_cot(x_g)

            x_self = att_self(x)
            if modal == 0:
                x_co = attn_co(x)
                x = x_COT_g + x_co * ( 1 / b) + x + x_self
            else:
                x = x_COT + x_self + x
            x_ff = ff(x)
            x =  x_ff + x



        return x



class ViT_patch(nn.Module):
    def __init__(self, *, image_size_h,image_size_w, patch_size_h,patch_size_w, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels, dim_head, dropout = 0., emb_dropout = 0.):
        super().__init__()
        # image_height, image_width = pair(image_size)
        image_height = image_size_h
        image_width = image_size_w

        # patch_height, patch_width = pair(patch_size)
        patch_height = patch_size_h
        patch_width = patch_size_w


        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        ).to('cuda')

        # self.to_patch_embedding_restore = nn.Sequential(
        #     Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width),
        #     nn.Linear(dim, patch_dim),
        # ).to('cuda')

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)).to('cuda')
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)).to('cuda')
        self.dropout = nn.Dropout(emb_dropout).to('cuda')
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout).to('cuda')

        self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)).to('cuda')
        self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)).to('cuda')
        self.spatial_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout).to('cuda')
        self.temporal_transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout).to('cuda')


        # self.co_transformer = Co_Transformer(dim, depth, heads, dim_head, mlp_dim, dropout).to('cuda')

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(

            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ).to('cuda')


    def forward(self, img, modal):
        ### VVIT
        x = self.to_patch_embedding(img) # [64, 54, 2048]
        b, n, _ = x.shape
        spa_cls_tokens = repeat(self.spatial_cls_token, '() n d -> b n d', b=b) # [64, 1, 2048]
        x = torch.cat((spa_cls_tokens, x), dim=1) # [64, 55, 2048]
        x_XXX =  self.pos_embedding


        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x) # [64, 55, 2048]
        x = self.spatial_transformer(x) # [64, 55, 2048]

        ### move sptial CLS token
        x_move = x[:, :(x.shape[1] - 1), :]

        #### fuse temporal CLS tokens
        temp_cls_tokens = repeat(self.temporal_cls_token, '() n d -> b n d', b=b)
        x_fuse_temp = torch.cat((temp_cls_tokens, x_move), dim=1)

        #### attent across Transformer
        x_attent = self.temporal_transformer(x_fuse_temp)

        x_mean = x_attent.mean(dim = 1) if self.pool == 'mean' else x_attent[:, 0]  # [64, 2048]
        x_out = self.to_latent(x_mean)
        trans_x = x_out
        mlp_x = self.mlp_head(x_out)

        return  trans_x, x_attent

