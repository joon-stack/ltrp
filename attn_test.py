import numpy as np

def softmax(x):
    # 안정적인 softmax 계산
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x)

np.set_printoptions(suppress=True, precision=6)

# 차원 설정
token_dim = 10  # 토큰 차원
num_tokens = 30  # 전체 토큰 개수
num_sampled = 15  # 샘플링할 토큰 개수

# 30개의 토큰을 정규분포에서 샘플링하여 생성 (각 토큰은 10차원)
np.random.seed(42)  # 재현성을 위해 시드 설정
X = np.random.normal(loc=0, scale=1, size=(num_tokens, token_dim))

# 앞의 15개 토큰만 선택
X_sampled = X[:num_sampled]

# 선형변환 행렬들 - 10x10 행렬을 정규분포에서 샘플링
np.random.seed(42)  # 동일한 가중치 행렬 사용
W_Q = np.random.normal(loc=0, scale=0.1, size=(token_dim, token_dim))
W_K = np.random.normal(loc=0, scale=0.1, size=(token_dim, token_dim))
W_V = np.random.normal(loc=0, scale=0.1, size=(token_dim, token_dim))

# 전체 30개 토큰에 대해 Q, K, V 계산
Q_full = X @ W_Q
K_full = X @ W_K
V_full = X @ W_V

# 샘플링된 15개 토큰에 대해 Q, K, V 계산
Q_sampled = X_sampled @ W_Q
K_sampled = X_sampled @ W_K
V_sampled = X_sampled @ W_V

print("Sampled Tokens - Q (shape):", Q_sampled.shape)
print(Q_sampled)
print("\nSampled Tokens - K (shape):", K_sampled.shape)
print(K_sampled)
print("\nSampled Tokens - V (shape):", V_sampled.shape)
print(V_sampled)

# 전체 토큰에 대한 Self-Attention 계산
dot_products_full = Q_full @ K_full.T  # (30,30)
d = Q_full.shape[1]  # d=10
scaled_dot_full = dot_products_full / np.sqrt(d)
attn_full = np.array([softmax(row) for row in scaled_dot_full])

# 샘플링된 토큰에 대한 Self-Attention 계산
dot_products_sampled = Q_sampled @ K_sampled.T  # (15,15)
scaled_dot_sampled = dot_products_sampled / np.sqrt(d)
attn_sampled = np.array([softmax(row) for row in scaled_dot_sampled])

print("\nAttention Weights (Sampled Tokens):")
print(attn_sampled)

# Attention output 계산 (전체 토큰)
attn_output_full = attn_full @ V_full  # (30,10)

# Attention output 계산 (샘플링된 토큰)
attn_output_sampled = attn_sampled @ V_sampled  # (15,10)

print("\nAttention Output (Full Tokens, first 15 rows):")
print(attn_output_full[:15])

print("\nAttention Output (Sampled Tokens):")
print(attn_output_sampled)

# 두 출력 간의 차이 계산 (앞의 15개 토큰만 비교)
difference = np.linalg.norm(attn_output_full[:15] - attn_output_sampled)
print(f"\n두 출력 간의 L2 거리: {difference:.6f}")

# 각 행에 대한 개별 차이 계산
row_differences = np.linalg.norm(attn_output_full[:15] - attn_output_sampled, axis=1)
print("\n각 행별 L2 거리:")
for i, diff in enumerate(row_differences):
    print(f"토큰 {i}: {diff:.6f}")
