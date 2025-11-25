import torch
import torch.nn as nn

class ZeroCopy(nn.Module):
    """
    Host-GPU Zero-Copy 방식
    - FPGA 사용 안 함
    - CPU 메모리에 weight 저장
    - GPU → CPU → GPU 경로로 unique indices 최적화
    """
    def __init__(self, num_embeddings:int, embedding_dim:int) -> None:
      super().__init__()
      self.num_embeddings = num_embeddings
      self.embedding_dim  = embedding_dim

      self.weight = nn.Parameter( # PyTorch가 학습 가능한 파라미터라고 인식
        torch.empty(num_embeddings, embedding_dim, device="cpu").pin_memory(),
        requires_grad=True
      )
    
    def reset_parameters(self) -> None:
       nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5) # 표준편차가 embedding_dim의 제곱근의 역수인 정규분포로 초기화

    def forward(self, indices:torch.LongTensor, scores: torch.Tensor)->torch.Tensor:
      return self._forward_optimized(indices, scores)
    
    def _forward_optimized(self, indices:torch.LongTensor, scores: torch.Tensor) -> torch.Tensor:
      """
      Optimize Unique Indices
      1. Eliminate duplicate at GPU
      2. Embedding Lookup at CPU
      3. Transfer to GPU
      4. GPU에서 원본 Shape 복원 + Weighted Sum
      """
      original_shape  = indices.shape                 #원래 shape를 저장
      flat_indices    = indices.flatten()             # 중복을 찾으려면, 1-Dimensiom으로 flat 필요
      unique_indices, inverse_indices = torch.unique(
        flat_indices,
        return_inverse=True
      )
      unique_cpu      = unique_indices.cpu()          # GPU-> CPU 전송
      unique_emb_cpu  = self.weight[unique_cpu]       # CPU에서 Indexing

      if unique_emb_cpu.dtype != torch.bfloat16:
          unique_emb_compressed = unique_emb_cpu.to(dtype=torch.bfloat16)
      else:
          unique_emb_compressed = unique_emb_cpu
      unique_emb_gpu = unique_emb_compressed.to(
        device=indices.device,                        # indices가 존재하는 GPU
        non_blocking=True                             # 비동기 전송?
      )
      # ========== Step 4: Recover shape + Weighted Sum ==========
      expanded_emb = unique_emb_gpu[inverse_indices]
      expanded_emb = expanded_emb.view(original_shape + (self.embedding_dim,))

      scores_expanded = scores.unsqueeze(-1)                # (B, N) -> (B, N, 1)
      output          = (expanded_emb * scores_expanded).sum(dim=1)  # Weighted Sum
      return output.to(scores.dtype)
