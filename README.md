# Memory Layers with CPU Pinned Memory

CPU Pinned Memory를 활용한 Memory Layer 구현체입니다. 원본 [Memory Layers at Scale](https://ai.meta.com/research/publications/memory-layers-at-scale/) 논문을 기반으로 하며, CPU 메모리에 대용량 임베딩을 저장하고 GPU에서 필요한 부분만 가져오는 방식으로 GPU 메모리 사용량을 최소화합니다.

## 주요 특징

- **CPU Pinned Memory**: PyTorch의 pinned memory를 이용한 효율적인 CPU-GPU 전송
- **Unique Indices 최적화**: 중복 인덱스 제거로 전송량 최소화
- **대용량 메모리 레이어**: Product Key Memory 기반
- **최적화된 학습**: FP16 mixed precision, BFloat16 압축 전송

## 프로젝트 구조

```
📦memory
 ┣ 📂lingua                    # 핵심 라이브러리
 ┃ ┣ 📂product_key            # Memory layer 구현
 ┃ ┃ ┣ 📜memory.py            # Product Key Memory 메인 로직
 ┃ ┃ ┣ 📜zero_copy.py         # CPU Pinned Memory 구현
 ┃ ┃ ┗ 📜colwise_embeddingbag.py
 ┃ ┗ 📜transformer.py
 ┣ 📂apps/main                # 메인 학습 앱
 ┃ ┣ 📜train.py               # 학습 스크립트
 ┃ ┗ 📂configs                # 설정 파일들
 ┣ 📂setup                    # 환경 구성 스크립트
 ┣ 📂agi_test                 # 실험 설정 및 스크립트
 ┗ 📂tokenizer                # 토크나이저
```

## Quick Start

### 1. 환경 구성

```bash
# 저장소 클론
git clone <repository-url>
cd memory

# Conda 환경 생성 및 패키지 설치
bash setup/create_env.sh

# 환경 활성화
conda activate prj_hs
```

**필수 요구사항:**
- Python 3.11
- CUDA 12.1
- PyTorch 2.5.0
- xformers

### 2. 데이터 준비

```bash
# Hugging Face 데이터셋 다운로드 및 준비
python setup/download_prepare_hf_data.py fineweb_edu_10bt <MEMORY_GB> \
    --data_dir ./data \
    --seed 42

# 토크나이저 다운로드 (Llama3)
python setup/download_tokenizer.py llama3 ./tokenizer \
    --api_key <HUGGINGFACE_TOKEN>
```

### 3. 학습 실행

#### 단일 GPU
```bash
CUDA_VISIBLE_DEVICES=0 python -m apps.main.train \
    config=agi_test/test_1/config/zero_copy.yaml
```

#### 실험 스크립트 사용
```bash
bash agi_test/test_1/script/zero_copy_train.sh
```

## 설정 가이드

주요 설정 파일: [agi_test/test_1/config/zero_copy.yaml](agi_test/test_1/config/zero_copy.yaml)

### CPU Pinned Memory 관련 설정

```yaml
model:
  productkey_args:
    zero_copy: true           # CPU Pinned Memory 활성화
    mem_offload: false        # FPGA offload 비활성화
    mem_n_keys: 8000          # 메모리 키 개수
    mem_share_values: true    # Value sharing 활성화
    mem_knn: 32               # K-nearest neighbors
    mem_k_dim: 512            # Key dimension
```

### 반드시 수정해야 할 경로

1. `data.root_dir`: 데이터 디렉토리 경로
2. `data.tokenizer.path`: 토크나이저 경로
3. `dump_dir`: 체크포인트 저장 경로 (스크립트에서 지정)

## 평가 (Evaluation)

```bash
# 체크포인트 평가
python -m apps.main.eval config=agi_test/test_1/config/eval_zero_copy.yaml
```

평가 태스크: HellaSwag, PIQA, NQ Open

## 핵심 구현

### CPU Pinned Memory Layer

메모리 값(values)을 GPU 메모리가 아닌 CPU pinned memory에 저장하여 GPU 메모리 사용량을 최소화합니다.

**주요 최적화 기법:**

1. **Unique Indices**: 중복 인덱스 제거로 CPU-GPU 전송량 감소
2. **Pinned Memory**: CPU 메모리를 pin하여 빠른 전송
3. **BFloat16 압축**: 전송 시 BFloat16으로 압축하여 대역폭 절약
4. **Non-blocking Transfer**: 비동기 전송으로 대기 시간 최소화

```python
# lingua/product_key/zero_copy.py
class ZeroCopy(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        # CPU pinned memory에 값 저장
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device="cpu").pin_memory()
        )

    def forward(self, indices, scores):
        # 1. GPU에서 중복 인덱스 제거
        unique_indices, inverse = torch.unique(indices.flatten(), return_inverse=True)
        # 2. CPU에서 임베딩 lookup
        unique_emb_cpu = self.weight[unique_indices.cpu()]
        # 3. BFloat16로 압축하여 GPU로 전송
        unique_emb_gpu = unique_emb_cpu.to(dtype=torch.bfloat16, device=indices.device)
        # 4. 원본 shape 복원 및 weighted sum
        return weighted_sum(unique_emb_gpu[inverse], scores)
```

## 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `mem_n_keys` | 8000 | 메모리 키 개수 |
| `mem_knn` | 32 | 검색할 nearest neighbors |
| `mem_k_dim` | 512 | 키 임베딩 차원 |
| `mem_heads` | 2 | 메모리 헤드 수 |
| `batch_size` | 2 | 배치 크기 |
| `seq_len` | 4096 | 시퀀스 길이 |

## 문제 해결

### CUDA Out of Memory
- `batch_size` 줄이기
- `mem_n_keys` 줄이기
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 설정

## 원본 논문 및 코드

이 코드는 [Meta Lingua](https://github.com/facebookresearch/lingua)를 기반으로 하며, zero-copy 기능이 추가되었습니다.

## Citation

```
@misc{memory_layers_scale,
  author = {Vincent-Pierre Berges, Barlas Oguz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, Gargi Gosh},
  title = {Memory Layers at Scale},
  url = {https://github.com/facebookresearch/memory},
  year = {2024}
}
```

## License

CC-BY-NC license
