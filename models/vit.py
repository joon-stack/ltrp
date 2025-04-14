# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py
import torch
from torch import nn
from einops import rearrange, repeat

import matplotlib.pyplot as plt
import seaborn as sns
import os

# helpers
NUM_FRAMES = 1
NUM_PATCHES = 1

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i+1) + [zeros] * (nwindow - i-1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask

def generate_merged_mask_matrix(npatches, nwindow):
    total_patches = sum(npatches)  # 30
    total_size = total_patches * nwindow  # 60
    mask = torch.zeros(total_size, total_size)  # (60, 60)

    # 각 그룹별로 처리
    start_idx = 0
    for size in npatches:
        group_size = size * nwindow  # 그룹 내 총 토큰 수
        frame_size = size  # 한 프레임의 패치 수
        base_block = torch.ones(frame_size, frame_size)  # 기본 블록 (모두 1)
        for i in range(nwindow):
            row_start = start_idx + i * frame_size
            for j in range(i + 1):  # 현재 프레임까지 (과거 포함)
                col_start = start_idx + j * frame_size
                mask[row_start:row_start + frame_size, col_start:col_start + frame_size] = base_block
        start_idx += group_size

    return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, 60, 60)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        if type(NUM_PATCHES) == list:
            self.bias = generate_merged_mask_matrix(NUM_PATCHES, NUM_FRAMES).to('cuda')
        else:
            self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to('cuda')
       


    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # apply causal mask
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # print("INFO: self.bias.shape", self.bias.shape, self.bias[0][0])
        # print("INFO: dots:", dots[0][0])
        

        attn = self.attend(dots)
        # print("INFO: attn.shape", attn.shape, attn.max(), attn.min())
        if attn.shape[0] == 1:
            save_dir = '/home/s2/youngjoonjeong/github/dino_wm/misc/attn_test'
            attn_vis = attn[0][0].cpu().detach().numpy() 
            # 196개의 각 row에 대해 처리
            for idx in range(attn_vis.shape[0]):
                attn_test = attn_vis[:, idx].reshape(14, 14)  # [196] -> [14, 14]
                
                # 히트맵 시각화
                plt.figure(figsize=(10, 8))
                sns.heatmap(attn_test,
                        cmap='viridis',
                        square=True,
                        annot=False,
                        cbar=True)
                
                plt.xlabel('Column')
                plt.ylabel('Row')
                plt.title(f'Attention Heatmap {idx}')
                
                # 파일 저장 (인덱스 포함된 이름으로)
                save_path = os.path.join(save_dir, f'attn_heatmap_{idx:03d}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()  # 메모리 관리
                
            print(f"196 heatmaps saved at: {save_dir}")

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)
    
class ViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames * (num_patches), dim)) # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool


    def forward(self, x, cluster_labels=None): # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, dim = x.shape
        _, num_full_patches, _ = self.pos_embedding.shape
        if cluster_labels is not None: # cluster_labels: (b, window_size, num_patches)
            F = self.num_frames 
            P = num_full_patches // F
            # print("INFO: cluster_labels.shape", cluster_labels.shape, cluster_labels)
            y = torch.unique(cluster_labels[0], return_counts=True)
            # print("INFO: cluster_labels.unique", y)
            num_clusters = cluster_labels.max().item() + 1
            # assert num_clusters == , "cluster_labels should have the same size as the number of patches"
            cluster_labels = cluster_labels.view(b, -1).to(torch.int64).cuda() # (b, num_patches*window_size)
            one_hot = torch.zeros(b, F*P, num_clusters).cuda()
            one_hot.scatter_(2, cluster_labels.unsqueeze(2), 1)
            one_hot = rearrange(one_hot, 'b n k -> b k n')
            pos_embedding_expanded = self.pos_embedding.expand(b, F*P, dim)
            # print("INFO: one_hot.shape", one_hot.shape)
            # print("INFO: pos_embedding_expanded.shape", pos_embedding_expanded.shape)
            sums = torch.bmm(one_hot, pos_embedding_expanded) # (b, num_clusters, dim)
            counts = one_hot.sum(dim=2) # (b, num_clusters)
            # print("INFO: counts: ", counts)
            # print("INFO: sums: ", torch.sum(sums[0], dim=1))
            # print("INFO: sums.shape, counts.shape", sums.shape, counts.shape)
            # print("INFO: counts: ", counts)
            # assert torch.sum(sums[1]) == torch.sum(self.pos_embedding[0]), f"sums: {torch.sum(sums[1])}, pos_embedding: {torch.sum(self.pos_embedding[0])}"
            # 수정된 assert 문
            pos_embedding_avg = sums / counts.unsqueeze(2)
            
            # print("INFO: x.shape, pos_embedding_avg.shape", x.shape, pos_embedding_avg.shape)
            x = x + pos_embedding_avg
            
        else:
            x = x + self.pos_embedding[:, :n]
        x = self.dropout(x) 
        x = self.transformer(x) 
        return x

    
class ObjectViTPredictor(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        if type(num_patches) == int:
            NUM_PATCHES = num_patches
            self.num_patches = num_patches
        else:
            NUM_PATCHES = list(num_patches)
            self.num_patches = list(num_patches)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim)) 
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        
        self.num_frames = num_frames

    def forward(self, x): # x: (b, window_size * H/patch_size * W/patch_size, 384)
       
        b, n, _ = x.shape
        if type(self.num_patches) == list:
            frame_pe = repeat(self.pos_embedding, '1 f d -> 1 (f p) d', p=sum(self.num_patches))
        else:
            frame_pe = repeat(self.pos_embedding, '1 f d -> 1 (f p) d', p=self.num_patches)
        x = x + frame_pe[:, :n]
        x = self.dropout(x) 
        x = self.transformer(x) 
        return x
    
class ViTPredictorWithoutPE(nn.Module):
    def __init__(self, *, num_patches, num_frames, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # update params for adding causal attention masks
        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, dim)) # dim for the pos encodings
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.pool = pool
        self.num_patches = num_patches
        self.num_frames = num_frames

        print("INFO: VITPredictorWithoutPE initialized with num_patches, self.pos_embedding: ", num_patches, self.pos_embedding.shape)

    def forward(self, x): # x: (b, window_size * H/patch_size * W/patch_size, 384)
        b, n, _ = x.shape
        frame_pe = repeat(self.pos_embedding, '1 f d -> 1 (f p) d', p=self.num_patches)

        x = x + frame_pe[:, :n]
        x = self.dropout(x) 
        x = self.transformer(x) 
        return x

# class