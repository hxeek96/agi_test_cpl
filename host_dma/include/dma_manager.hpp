#ifndef DMA_MANAGER_HPP
#define DMA_MANAGER_HPP

#include <cuda_runtime.h>
#include <sys/ioctl.h>
#include <linux/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <cstdint>
#include <iostream>

// Define IOCTL Command
#define IOCTL_XDMA_DIR_TO_HOST  _IO('q',0)
#define IOCTL_XDMA_DIR_TO_GPU   _IO('q',7)

// XDMA DEVICE
#define DEVICE_NAME_H2C "/dev/xdma0_h2c_0"
#define DEVICE_NAME_C2H "/dev/xdma0_c2h_0"

// Transfer Max Size
#define RW_MAX_SIZE     0x7ffff000

#define CHUNK_SIZE_MB 220
#define CHUNK_SIZE (1ULL * 1024ULL * 1024ULL * CHUNK_SIZE_MB / sizeof(float))

// Error Handling Macro
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error " << __FILE__ << ":" << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA Error"); \
    } \
} while(0)

class dma_manager{
	private: // class's member function can access
		int 		fd_h2c;
		int 		fd_c2h;
		float 	*host_buffer;
		size_t 	buffer_size;
		bool 		initialized;
		void* 	reg_base_addr;
		void 		set_device_mode(int fd, int ioctl_cmd, const char* mode_name);
	public:
		dma_manager();
		~dma_manager();

		// initialize
		bool initialize(size_t table_size_bytes);
		bool set_gpu_mode();
		bool set_host_mode();
		/*
			const char *fname : 파일 이름, 에러 메시지를 기록
			int fd : open()을 통해 얻은 파일 디스크립터
			char *buffer : Host 측, User 공간 메모리 버퍼 주소 => host_buff
				Read: FPGA->HOST로 읽은 데이터가 해당 주소에 채워짐.
				WRITE: HOST에서 해당 주소의 데이터를 FPGA 쪽으로 전송.
			uint64_t size : 총 몇 바이트를 Read/Write 할지
			uint64_t base : FPGA 측, 주소(디바이스 파일의 내부 주소 공간 offset) =>fpga_offset
		*/
		bool host_to_fpga(const char* host_buff, uint64_t size, uint64_t fpga_offset);
		bool fpga_to_host(char* host_buff, uint64_t size, uint64_t fpga_offset);
		bool gpu_to_fpga(char* gpu_buff, uint64_t size, uint64_t fpga_offset);
		bool fpga_to_gpu(char* gpu_buff, uint64_t size, uint64_t fpga_offset);
		


		bool host_to_fpga(const char* fname, int fd, const char* host_buff, uint64_t size, uint64_t fpga_offset);
		bool fpga_to_host(const char* fname, int fd, char* host_buff, uint64_t size, uint64_t fpga_offset);
		bool gpu_to_fpga(const char* fname, int fd, char* gpu_buff, uint64_t size, uint64_t fpga_offset);
		bool fpga_to_gpu(const char* fname, int fd, char* gpu_buff, uint64_t size, uint64_t fpga_offset);
		
		bool initialize_register();
		void write_reg(void* base, uint32_t offset, uint32_t data);
		void write_reg_i(int i, uint32_t val);
		uint32_t read_reg(void* base, uint32_t offset);
		uint32_t read_reg_i(int i);
		void host_setcommand(uint32_t command, uint32_t data0, uint32_t data1);

		void cleanup();
			// read_only function-> just return the status of [initialized]
		bool is_initialized() const{ return initialized;}
};

extern "C" {
	void* dma_new();
	void dma_delete(void* ptr);
	bool dma_initialize(void* ptr, size_t table_size_bytes);
	bool dma_host_to_fpga(void* ptr, float* data, uint64_t size, uint64_t fpga_offset);
	bool dma_fpga_to_host(void* ptr, float* data, uint64_t size, uint64_t fpga_offset);
	bool dma_host_to_fpga_fd(void* ptr, const char* fname, int fd, float* data, uint64_t size, uint64_t fpga_offset);
	bool dma_fpga_to_host_fd(void* ptr, const char* fname, int fd, float* data, uint64_t size, uint64_t fpga_offset);
	bool dma_is_initialized(void* ptr);
	bool dma_cleanup(void* ptr);
	bool dma_set_gpu_mode(void* ptr);
	bool dma_set_host_mode(void* ptr);
	bool dma_gpu_to_fpga(void* ptr, void* gpu_ptr, uint64_t size, uint64_t fpga_offset);
	bool dma_fpga_to_gpu(void* ptr, void* gpu_ptr, uint64_t size, uint64_t fpga_offset);
	bool dma_gpu_to_fpga_fd(void* ptr, const char* fname, int fd, void* gpu_ptr, uint64_t size, uint64_t fpga_offset);
	bool dma_fpga_to_gpu_fd(void* ptr, const char* fname, int fd, void* gpu_ptr, uint64_t size, uint64_t fpga_offset);
	bool dma_initialize_register(void* ptr);
	
	void dma_host_setcommand(void* ptr, uint32_t command, uint32_t data0, uint32_t data1);
	void dma_write_reg_i(void* ptr, int i, uint32_t val);
	uint32_t dma_read_reg_i(void* ptr, int i);
}

#endif