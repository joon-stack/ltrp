import argparse
import torch
from PIL import Image
from factory import get_score_net  # Score Net 모델 가져오기
from utils.datasets import build_transform  # 이미지 전처리 변환
import cv2
import numpy as np
import os

# 1. 커맨드 라인 인자 파싱 함수
def get_args_parser():
    parser = argparse.ArgumentParser(description='Get Patch Ranking from Score Net', add_help=False)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', 
                        choices=['cuda', 'cpu'], help='Device to use (cuda or cpu)')
    parser.add_argument('--input_size', default=224, type=int, help='Input image size')
    parser.add_argument('--score_net', default='ltrp_cluster_vs', type=str, 
                        help='Score Net model name (e.g., ltrp_cluster_vs)')
    parser.add_argument('--checkpoint_path', type=str, default=None, 
                        help='Path to the trained checkpoint file')
    parser.add_argument('--dataset_name', type=str, default='block_push',
                        help='Path to the input image')
    return parser

# 2. 패치 랭킹 얻는 함수
def get_patch_ranking(score_net, image_path, args):
    # 이미지 전처리
    transform = build_transform(is_train=False, args=args)  # 평가용 변환 생성
    image = Image.open(image_path).convert('RGB')      # 이미지 로드
    image = transform(image).unsqueeze(0).to(args.device)  # 배치 차원 추가 및 장치 이동
    print("INFO: image.min, image.max", image.min(), image.max())

    # Score Net을 평가 모드로 설정
    score_net.eval()

    # 패치 중요도 예측 (추론)
    with torch.no_grad():  # 기울기 계산 비활성화
        scores = score_net(image)

    # 패치 랭킹 생성
    _, indices = torch.sort(scores, dim=1, descending=True)  # 점수 기준 내림차순 정렬
    print("INFO: ", scores)
    return indices.cpu().numpy()  # numpy 배열로 변환하여 반환

def visualize_patch_ranking(image_path, ranking, patch_size=16, alpha=0.5):
    # 원본 이미지 로드
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)  # PIL -> numpy 배열
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB -> BGR로 변환

    # 이미지 크기와 패치 수 계산
    h, w, _ = image.shape
    grid_h = h // patch_size  # 세로 패치 수
    grid_w = w // patch_size  # 가로 패치 수
    num_patches = grid_h * grid_w

    # 랭킹을 중요도 점수로 변환
    scores = np.zeros(num_patches)
    for i, idx in enumerate(ranking[0]):  # ranking이 2D 배열이라 가정
        scores[idx] = num_patches - i  # 1위 패치에 높은 점수 부여

    # 점수를 0~1로 정규화
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    # 히트맵 생성 (패치 단위)
    heatmap = np.zeros((grid_h, grid_w))
    for i in range(grid_h):
        for j in range(grid_w):
            patch_idx = i * grid_w + j
            heatmap[i, j] = scores[patch_idx]

    # 히트맵을 이미지 크기로 리사이즈
    heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # 히트맵을 컬러맵으로 변환 (JET 컬러맵 사용)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 원본 이미지와 히트맵 오버레이
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    # 결과 이미지 저장
    cv2.imwrite('output_dir/overlay.png', overlay)

def visualize_patch_blocks(image_path, ranking, args, patch_size=16, alpha=0.5):
    # 이미지 로드
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)  # PIL -> numpy 배열
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # RGB -> BGR로 변환

    # 이미지 크기와 패치 수 계산
    h, w, _ = image.shape
    grid_h = h // patch_size  # 세로 패치 수
    grid_w = w // patch_size  # 가로 패치 수
    num_patches = grid_h * grid_w

    # 랭킹을 점수로 변환 (중요도 반영)
    scores = np.zeros(num_patches)
    for i, idx in enumerate(ranking[0]):  # ranking이 2D 배열이라 가정
        scores[idx] = num_patches - i  # 높은 순위에 높은 점수 부여

    # 점수를 0~1로 정규화
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    # 패치 단위로 색상 블록 생성
    heatmap = np.zeros((h, w))
    for i in range(grid_h):
        for j in range(grid_w):
            patch_idx = i * grid_w + j
            heatmap[i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size] = scores[patch_idx]

    # 색상 적용 (JET 컬러맵 사용)
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # 원본 이미지와 오버레이
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap_color, alpha, 0)

    fname = image_path.split('/')[-1]
    dirname = os.path.dirname(image_path)

    # 결과 저장
    if args.checkpoint_path:
        cv2.imwrite(f'{dirname}/patch_blocks_{fname}', overlay)
    else:
        cv2.imwrite(f'{dirname}/patch_blocks_random_{fname}', overlay)
    print(f"패치 블록이 적용된 이미지가 patch_blocks_{fname}로 저장되었습니다.")


if __name__ == '__main__':
    # 1. 인자 파싱
    parser = get_args_parser()
    args = parser.parse_args()

    # 2. Score Net 모델 로드
    score_net = get_score_net(args)  # 모델 초기화
    if args.checkpoint_path:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')  # 체크포인트 로드
        # 체크포인트에서 score_net 관련 키만 추출
        score_net_state_dict = {}
        for key, value in checkpoint['model'].items():
            if key.startswith('score_net.'):
                new_key = key[len('score_net.'):]  # 'score_net.' 접두사 제거
                score_net_state_dict[new_key] = value

        # Score Net에 필터링된 상태 로드
        score_net.load_state_dict(score_net_state_dict)
    
    score_net.to(args.device)  # 모델을 설정한 장치로 이동

    image_paths = [f'output_dir/{args.dataset_name}/x{i}_image.png' for i in range(5)]  # 이미지 경로 리스트
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        # 3. 패치 랭킹 얻기
        ranking = get_patch_ranking(score_net, image_path, args)

        # 4. 시각화
        visualize_patch_blocks(image_path, ranking, args)

        # 5. 결과 출력
        print("Patch Ranking:", ranking)
    