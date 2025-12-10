import torch
import torch.nn as nn
import numpy as np
import sys
import time

sys.path.append('/home/hs/agi_test_cpl/host_dma/src/py')
from dma_binding import DMA


DDR0_BASE       = 0x0000_0000_0000_0000  # Indices
DDR1_BASE       = 0x0000_0004_0000_0000  # Embedding table base
DDR1_SRC_BASE   = DDR1_BASE          # Source table

VALUE_TABLE_SIZE = 1024 * 1024 * 1024 * 2                   # 2GB
DDR1_DST_BASE    = DDR1_BASE + VALUE_TABLE_SIZE + 0X1000    # Destination

class OffloadFpga(nn.Module):
    """
    FPGA Offloading -> Product Key Memory Values
    - FPGA DDR store Weight(Tensor)
    - GPU send Indices to FPGA(forward)
    - FPGA find the Tensor in DDR1, and Gather the Tensor to Optimized DMA Transfer
    """
    def __init__(self, num_embeddings:int, embedding_dim:int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim  = embedding_dim

        # 1. Generate weight parameter (bfloat16) at CPU
        self.weight = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim, dtype=torch.bfloat16),
            requires_grad=False
        )

        # 2. Initialize DMA
        self.dma = DMA()
        table_size_bytes = num_embeddings * embedding_dim * 2  # bfloat16 = 2 bytes
        if not self.dma.initialize(table_size_bytes):
            raise RuntimeError("Failed to initialize DMA")
        if not self.dma.initialize_register():
            raise RuntimeError("Failed to initialize DMA register")

        # 3. Transfer weight to FPGA DDR1
        self._transfer_weight_to_fpga(self.weight)

    def _transfer_weight_to_fpga(self, weight_bf16: torch.Tensor):
        weight_uint16   = weight_bf16.view(torch.uint16)
        success         = self.dma.host_to_fpga(weight_uint16, fpga_offset=DDR1_SRC_BASE)
        if not success:
            raise RuntimeError("Failed to Weight -> FPGA")
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Checkpoint 로드 시 자동으로 호출되는 훅
        weight가 업데이트되면 자동으로 FPGA로 재전송
        """
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, 
            missing_keys, unexpected_keys, error_msgs
        )
        
        # Checkpoint에서 weight가 로드되었으므로 FPGA로 재전송
        self._transfer_weight_to_fpga(self.weight)
    
    def _trigger_fpga_gather(self):
        self.dma.write_reg_i(0, 0x1)
    
    def _wait_for_completion(self, timeout_ms=5000):
        start_time = time.time()
        while True:
            status  = self.dma.read_reg_i(1)
            op_done = (status >> 2) & 0x1
            if op_done:
                return
            
            elapsed = (time.time() - start_time) * 1000
            if elapsed > timeout_ms:
                raise TimeoutError(
                    f"FPGA timeout after {timeout_ms}ms (status=0x{status:08X})"
                )
            time.sleep(0.001)
    
    def forward(self, indices: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        batch_size  = indices.shape[0]
        num_indices = indices.shape[1]
        
        # 1. Indices -> FPGA DDR0 (P2P)
        if not indices.is_cuda:
            indices = indices.cuda()
        
        if not self.dma.gpu_to_fpga(indices, fpga_offset=DDR0_BASE):
            raise RuntimeError("Failed to transfer indices to FPGA")

        # 2. Trigger FPGA gather
        self._trigger_fpga_gather()

        # 3. Wait for completion
        self._wait_for_completion()

        # 4. Read result (FPGA DDR1 -> GPU)
        result_shape = (batch_size * num_indices, self.embedding_dim)
        result_elements = batch_size * num_indices * self.embedding_dim
        result_bytes = result_elements * 2  # bfloat16 = 2 bytes
        
        # 최소 32MB로 할당 (P2P 호환성)
        MIN_SIZE_BYTES = 32 * 1024 * 1024  # 32MB
        
        # checkpoint의 dtype 사용 (fp16 or bfloat16)
        weight_dtype = self.weight.dtype
        
        if result_bytes < MIN_SIZE_BYTES:
            # 패딩 추가 (P2P 전송을 위한 최소 크기)
            total_elements = MIN_SIZE_BYTES // 2
            result_padded = torch.zeros(total_elements, dtype=weight_dtype, device='cuda')
            result = result_padded[:result_elements].view(result_shape)
        else:
            result = torch.zeros(result_shape, dtype=weight_dtype, device='cuda')
        
        if not self.dma.fpga_to_gpu(result, fpga_offset=DDR1_DST_BASE):
            raise RuntimeError("Failed to read result from FPGA")

        # 5. Weighted sum (GPU)
        result = result.view(batch_size, num_indices, self.embedding_dim)
        
        if not scores.is_cuda:
            scores = scores.cuda()
        scores = scores.to(result.dtype)
        
        # (B, N, D) * (B, N, 1) → sum(dim=1) → (B, D)
        scores_expanded = scores.unsqueeze(-1)
        output = (result * scores_expanded).sum(dim=1)
        
        return output