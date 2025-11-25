import ctypes
import numpy as np
import torch
from typing import Optional, Union # for type hinting

class DMA:
    def __init__(self):
        self.lib = ctypes.CDLL('/home/hs/farnn/memory/host_dma/libdma_manager.so')
        
        self._setup_function_signatures()

        self.ptr = self.lib.dma_new()
        if not self.ptr:
            raise RuntimeError("Failed to create DMA Transfer object")
    
    def _setup_function_signatures(self):
        # 생성자
        self.lib.dma_new.argtypes = []
        self.lib.dma_new.restype  = ctypes.c_void_p
        # 소멸자
        self.lib.dma_delete.argtypes = [ctypes.c_void_p]
        self.lib.dma_delete.restype  = None
        # 초기화
        self.lib.dma_initialize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.dma_initialize.restype  = ctypes.c_bool
        # Host -> FPGA
        self.lib.dma_host_to_fpga.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_uint64]
        self.lib.dma_host_to_fpga.restype  = ctypes.c_bool
        # FPGA -> Host
        self.lib.dma_fpga_to_host.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_uint64]
        self.lib.dma_fpga_to_host.restype  = ctypes.c_bool
        # Host -> FPGA (with FD)
        self.lib.dma_host_to_fpga_fd.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_uint64]
        self.lib.dma_host_to_fpga_fd.restype  = ctypes.c_bool
        # FPGA -> Host (with FD)
        self.lib.dma_fpga_to_host_fd.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.c_uint64]
        self.lib.dma_fpga_to_host_fd.restype  = ctypes.c_bool
        # 초기화 상태 확인
        self.lib.dma_is_initialized.argtypes = [ctypes.c_void_p]
        self.lib.dma_is_initialized.restype  = ctypes.c_bool
        # 정리
        self.lib.dma_cleanup.argtypes = [ctypes.c_void_p]
        self.lib.dma_cleanup.restype  = ctypes.c_bool

        # 모드 전환
        self.lib.dma_set_gpu_mode.argtypes = [ctypes.c_void_p]
        self.lib.dma_set_gpu_mode.restype  = ctypes.c_bool
        self.lib.dma_set_host_mode.argtypes = [ctypes.c_void_p]
        self.lib.dma_set_host_mode.restype  = ctypes.c_bool

        # GPU 포인터 전송 (내부 fd)
        self.lib.dma_gpu_to_fpga.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        self.lib.dma_gpu_to_fpga.restype  = ctypes.c_bool
        self.lib.dma_fpga_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        self.lib.dma_fpga_to_gpu.restype  = ctypes.c_bool

        # GPU 포인터 전송 (FD 오버로드)
        self.lib.dma_gpu_to_fpga_fd.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        self.lib.dma_gpu_to_fpga_fd.restype  = ctypes.c_bool
        self.lib.dma_fpga_to_gpu_fd.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
        self.lib.dma_fpga_to_gpu_fd.restype  = ctypes.c_bool

        self.lib.dma_initialize_register.argtypes = [ctypes.c_void_p]
        self.lib.dma_initialize_register.restype  = ctypes.c_bool

        self.lib.dma_host_setcommand.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
        self.lib.dma_host_setcommand.restype = None

        self.lib.dma_write_reg_i.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32]
        self.lib.dma_write_reg_i.restype = None

        self.lib.dma_read_reg_i.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.dma_read_reg_i.restype  = ctypes.c_uint32

    def __del__(self):
        if hasattr(self, 'ptr') and self.ptr:
            self.lib.dma_delete(self.ptr)

    def initialize(self, table_size_bytes: int) ->bool:
        # Initialize DMA Transfer
        return self.lib.dma_initialize(self.ptr, table_size_bytes)

    def set_gpu_mode(self) -> bool:
        return self.lib.dma_set_gpu_mode(self.ptr)
    def set_host_mode(self) -> bool:
        return self.lib.dma_set_host_mode(self.ptr)
    
    def gpu_to_fpga(self, tensor: torch.Tensor, fpga_offset: int = 0) -> bool:
        assert isinstance(tensor, torch.Tensor) and tensor.is_cuda, "Need CUDA Tensor"
        t = tensor.contiguous()
        size = t.element_size() * t.numel()
        return self.lib.dma_gpu_to_fpga(self.ptr, ctypes.c_void_p(t.data_ptr()), size, fpga_offset)
    
    def fpga_to_gpu(self, tensor: torch.Tensor, fpga_offset: int = 0) -> bool:
        assert isinstance(tensor, torch.Tensor) and tensor.is_cuda, "CUDA Tensor 필요"
        t = tensor.contiguous()
        size = t.element_size() * t.numel()
        return self.lib.dma_fpga_to_gpu(self.ptr, ctypes.c_void_p(t.data_ptr()), size, fpga_offset)
    
    def host_to_fpga(self, data: Union[torch.Tensor, np.ndarray], fpga_offset: int=0)->bool:
        """KV 테이블을 FPGA로 전송 (내부 fd 사용)"""
        if isinstance(data, torch.Tensor): #data가 Pytorch Tensor 인지 검사
            data_np = data.cpu().numpy()
        else:
            data_np = data
        return self.lib.dma_host_to_fpga(
            self.ptr,
            data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            data_np.nbytes,
            fpga_offset
        )

    def fpga_to_host(self, data: Union[torch.Tensor, np.ndarray], fpga_offset: int = 0) -> bool:
        """FPGA에서 KV Table을 Host로 읽어옴"""
        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            data_np = data.cpu().numpy()
        else:
            data_np = data
        
        # C++ 함수 호출
        result = self.lib.dma_fpga_to_host(
            self.ptr,
            data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            data_np.nbytes,
            fpga_offset
        )
        
        # Torch인 경우 원본에 복사
        if is_torch:
            data.copy_(torch.from_numpy(data_np))
        
        return result

    def host_to_fpga_fd(self, fname: str, fd:int, data: Union[torch.Tensor, np.ndarray], fpga_offset: int = 0)->bool:
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data
        return self.lib.dma_host_to_fpga_fd(
            self.ptr, 
            fname.encode('utf-8'),
            fd,
            data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            data_np.nbytes,
            fpga_offset
        )
    
    def fpga_to_host_fd(self, fname: str, fd: int, data: Union[torch.Tensor, np.ndarray], fpga_offset: int = 0) -> bool:
        if isinstance(data, torch.Tensor):
            data_np = data.cpu().numpy()
        else:
            data_np = data
        return self.lib.dma_fpga_to_host_fd(
            self.ptr,
            fname.encode('utf-8'),
            fd,
            data_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            data_np.nbytes,
            fpga_offset
        )
    def gpu_to_fpga_fd(self, fname: str, fd: int, tensor: torch.Tensor, fpga_offset: int = 0) -> bool:
        assert isinstance(tensor, torch.Tensor) and tensor.is_cuda
        t = tensor.contiguous()
        size = t.element_size() * t.numel()
        return self.lib.dma_gpu_to_fpga_fd(self.ptr, fname.encode("utf-8"), fd, ctypes.c_void_p(t.data_ptr()), size, fpga_offset)

    def fpga_to_gpu_fd(self, fname: str, fd: int, tensor: torch.Tensor, fpga_offset: int = 0) -> bool:
        assert isinstance(tensor, torch.Tensor) and tensor.is_cuda
        t = tensor.contiguous()
        size = t.element_size() * t.numel()
        return self.lib.dma_fpga_to_gpu_fd(self.ptr, fname.encode("utf-8"), fd, ctypes.c_void_p(t.data_ptr()), size, fpga_offset)   

    def initialize_register(self) -> bool:
        return self.lib.dma_initialize_register(self.ptr)
    
    def host_setcommand(self, command:int, data0:int, data1:int)->None:
        return self.lib.dma_host_setcommand(self.ptr, command, data0, data1)
    
    def write_reg_i(self, reg_index: int, value: int) ->None:
        return self.lib.dma_write_reg_i(self.ptr, reg_index, value)
    
    def read_reg_i(self, reg_index:int)->int:
        return self.lib.dma_read_reg_i(self.ptr, reg_index)

    def is_initialized(self) ->bool:
        return self.lib.dma_is_initialized(self.ptr)
    
    def cleanup(self):
        self.lib.dma_cleanup(self.ptr)