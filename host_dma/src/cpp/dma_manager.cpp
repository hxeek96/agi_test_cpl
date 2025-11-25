#include "../../include/dma_manager.hpp"
#include "../../include/fpga_defines.hpp" 
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <iostream>
#include <cstring>
#include <sys/types.h>
#include <algorithm>
#include <sys/mman.h>
#include <unistd.h>

dma_manager::dma_manager()
    : fd_h2c(-1), fd_c2h(-1), host_buffer(nullptr), 
      buffer_size(0), initialized(false), reg_base_addr(nullptr) {  // 추가
}
dma_manager::~dma_manager() {
    cleanup();
}

/**
 * Initialize DMA manager with specified buffer size
 * @param table_size_bytes Size of host buffer in bytes
 * @return true on success, false on failure
 */
bool dma_manager::initialize(size_t table_size_bytes){
	try{
		// Open H2C (Host to Card) device
		fd_h2c = open(DEVICE_NAME_H2C, O_RDWR);
		if(fd_h2c < 0){
			std::cerr << "Failed to open H2C device: " << strerror(errno) << std::endl;
			return false;
		}

		// Open C2H (Card to Host) device
		fd_c2h = open(DEVICE_NAME_C2H, O_RDWR);
		if(fd_c2h < 0){
			std::cerr << "Failed to open C2H device: " << strerror(errno) << std::endl;
			close(fd_h2c);
			return false;
		}

		// Allocate host memory buffer
		host_buffer = new float[table_size_bytes / sizeof(float)];
		if(!host_buffer){
			std::cerr << "Failed to allocate host memory" << std::endl;
			cleanup();
			return false;
		}

		buffer_size = table_size_bytes;
		initialized = true;

		std::cout << "DMA Transfer initialized with " << (table_size_bytes / (1024*1024)) << " MB buffer" << std::endl;
		return true;
	} catch(const std::exception& e){
		std::cerr << "DMA initialization failed: " << e.what() << std::endl;
		cleanup();
		return false;
	}
}

/**
 * Transfer data from host buffer to FPGA memory (low-level implementation)
 * @param fname Channel name for logging
 * @param fd File descriptor for DMA channel
 * @param host_buff Source buffer pointer
 * @param size Transfer size in bytes
 * @param fpga_offset FPGA memory offset
 * @return true on success, false on failure
 */
 bool dma_manager::host_to_fpga(const char* fname, int fd, const char* host_buff, uint64_t size, uint64_t fpga_offset){
	if(!initialized){
			std::cerr << "DMA transfer not initialized" << std::endl;
			return false;
	}
	if(!set_host_mode()){
		std::cerr << "Failed to set host mode" << std::endl;
		return false;
	}
	const uint64_t MAX_CHUNK = 2ULL * 1024 * 1024 * 1024;  // 2GB
	uint64_t total_written = 0;
	
	while(total_written < size) {
			uint64_t chunk_size = std::min(MAX_CHUNK, size - total_written);
			uint64_t current_offset = fpga_offset + total_written;
			
			std::cout << "Chunk: seeking to 0x" << std::hex << current_offset << std::dec 
								<< ", size: " << (chunk_size / (1024*1024)) << " MB" << std::endl;
			
			// Seek for this chunk
			off64_t seek_result = lseek64(fd, current_offset, SEEK_SET);
			if(seek_result != static_cast<off64_t>(current_offset)){
					std::cerr << "Failed to seek to offset 0x" << std::hex << current_offset << std::dec << std::endl;
					return false;
			}
			
		// Write this chunk
		uint64_t chunk_written = 0;
		const char* chunk_buf = host_buff + total_written;
		
		while(chunk_written < chunk_size){
				uint64_t bytes = chunk_size - chunk_written;
				if(bytes > RW_MAX_SIZE){
						bytes = RW_MAX_SIZE;
				}

				// Seek before each write to prevent file pointer wrap-around at 2GB boundary
				uint64_t write_offset = current_offset + chunk_written;
				if(lseek64(fd, write_offset, SEEK_SET) != static_cast<off64_t>(write_offset)){
						std::cerr << "Failed to seek to write offset 0x" << std::hex << write_offset << std::dec << std::endl;
						return false;
				}

				ssize_t result = write(fd, chunk_buf, bytes);
				if(result != static_cast<ssize_t>(bytes)){
						std::cerr << "Failed to write: " << strerror(errno) << std::endl;
						return false;
				}

				chunk_written += bytes;
				chunk_buf += bytes;
		}
			
			total_written += chunk_size;
	}

	return true;
}

bool dma_manager::host_to_fpga(const char* host_buff, uint64_t size, uint64_t fpga_offset){
	return host_to_fpga("H2C", fd_h2c, host_buff, size, fpga_offset);
}

bool dma_manager::fpga_to_host(char* host_buff, uint64_t size, uint64_t fpga_offset){
	return fpga_to_host("C2H", fd_c2h, host_buff, size, fpga_offset);
}

/**
 * Transfer data from FPGA memory to host buffer (low-level implementation)
 * @param fname Channel name for logging
 * @param fd File descriptor for DMA channel
 * @param host_buff Destination buffer pointer
 * @param size Transfer size in bytes
 * @param fpga_offset FPGA memory offset
 * @return true on success, false on failure
 */
 bool dma_manager::fpga_to_host(const char* fname, int fd, char* host_buff, uint64_t size, uint64_t fpga_offset){
	if(!initialized){
			std::cerr << "DMA transfer not initialized" << std::endl;
			return false;
	}
	if(!set_host_mode()){
		std::cerr << "Failed to set host mode" << std::endl;
		return false;
	}
	const uint64_t MAX_CHUNK = 2ULL * 1024 * 1024 * 1024;  // 2GB
	uint64_t total_read = 0;
	
	while(total_read < size) {
			uint64_t chunk_size = std::min(MAX_CHUNK, size - total_read);
			uint64_t current_offset = fpga_offset + total_read;
			
			std::cout << "Read chunk: seeking to 0x" << std::hex << current_offset << std::dec 
								<< ", size: " << (chunk_size / (1024*1024)) << " MB" << std::endl;
			
			// Seek for this chunk
			off64_t seek_result = lseek64(fd, current_offset, SEEK_SET);
			if(seek_result != static_cast<off64_t>(current_offset)){
					std::cerr << "Failed to seek to offset 0x" << std::hex << current_offset << std::dec << std::endl;
					return false;
			}
			
		// Read this chunk
		uint64_t chunk_read = 0;
		char* chunk_buf = host_buff + total_read;
		
		while(chunk_read < chunk_size){
				uint64_t bytes = chunk_size - chunk_read;
				if(bytes > RW_MAX_SIZE){
						bytes = RW_MAX_SIZE;
				}

				// Seek before each read to prevent file pointer wrap-around at 2GB boundary
				uint64_t read_offset = current_offset + chunk_read;
				if(lseek64(fd, read_offset, SEEK_SET) != static_cast<off64_t>(read_offset)){
						std::cerr << "Failed to seek to read offset 0x" << std::hex << read_offset << std::dec << std::endl;
						return false;
				}

				ssize_t result = read(fd, chunk_buf, bytes);
				if(result != static_cast<ssize_t>(bytes)){
						std::cerr << "Failed to read: " << strerror(errno) << std::endl;
						return false;
				}

				chunk_read += bytes;
				chunk_buf += bytes;
		}
			
			total_read += chunk_size;
	}

	return true;
}
/**
 * Set DMA device mode using ioctl
 * @param fd File descriptor
 * @param ioctl_cmd IOCTL command
 * @param mode_name Mode description for error messages
 */
void dma_manager::set_device_mode(int fd, int ioctl_cmd, const char* mode_name) {
    if (ioctl(fd, ioctl_cmd) < 0) {
        throw std::runtime_error(std::string("IOCTL failed (") + mode_name + "): " + strerror(errno));
    }
}

/**
 * Configure DMA channels for host-to-host transfers
 * @return true on success, false on failure
 */
bool dma_manager::set_host_mode(){
	try{
		set_device_mode(fd_h2c, IOCTL_XDMA_DIR_TO_HOST, "H2C Host");
		set_device_mode(fd_c2h, IOCTL_XDMA_DIR_TO_HOST, "C2H Host");
		return true;
	} catch(const std::exception& e){
		std::cerr << "Failed to set host mode: " << e.what() << std::endl;
		return false;
	}
}

/**
 * Configure DMA channels for GPU direct transfers
 * @return true on success, false on failure
 */
bool dma_manager::set_gpu_mode(){
	try{
		set_device_mode(fd_h2c, IOCTL_XDMA_DIR_TO_GPU, "H2C GPU Direct");
		set_device_mode(fd_c2h, IOCTL_XDMA_DIR_TO_GPU, "C2H GPU Direct");
		return true;
	} catch(const std::exception& e){
		std::cerr << "Failed to set GPU mode: " << e.what() << std::endl;
		return false;
	}
}

bool dma_manager::gpu_to_fpga(char* gpu_buff, uint64_t size, uint64_t fpga_offset){
	return gpu_to_fpga("H2C-GPU", fd_h2c, gpu_buff, size, fpga_offset);
};
bool dma_manager::fpga_to_gpu(char* gpu_buff, uint64_t size, uint64_t fpga_offset){
	return fpga_to_gpu("C2H-GPU", fd_c2h, gpu_buff, size, fpga_offset);
};


/**
 * Transfer data from GPU memory to FPGA memory with chunking (low-level implementation)
 * Handles large transfers by breaking them into manageable chunks
 * @param fname Channel name for logging
 * @param fd File descriptor for DMA channel
 * @param gpu_buff GPU buffer pointer
 * @param size Total transfer size in bytes
 * @param fpga_offset FPGA memory offset
 * @return true on success, false on failure
 */
bool dma_manager::gpu_to_fpga(const char* fname, int fd, char* gpu_buff, uint64_t size, uint64_t fpga_offset){
	if(!initialized){
		std::cerr << "GPU-FPGA transfer not initialized" << std::endl;
		return false;
	}

	// Switch to GPU direct mode
	if(!set_gpu_mode()){
		std::cerr << "Failed to set GPU mode" << std::endl;
		return false;
	}

	// Calculate chunk parameters for large transfers
	uint64_t chunk_size_bytes = CHUNK_SIZE * sizeof(float);
	uint64_t total_chunks = (size + chunk_size_bytes - 1) / chunk_size_bytes;

	// Transfer data in chunks
	for (uint64_t chunk = 0; chunk < total_chunks; chunk++) {
		uint64_t current_offset = fpga_offset + (chunk * chunk_size_bytes);
		uint64_t current_chunk_size = std::min(chunk_size_bytes, size - (chunk * chunk_size_bytes));
		char* current_gpu_ptr = gpu_buff + (chunk * chunk_size_bytes);

		// Seek to FPGA offset for this chunk
		if (lseek64(fd, current_offset, SEEK_SET) != static_cast<off64_t>(current_offset)) {
			std::cerr << "Failed to seek to chunk offset " << current_offset << std::endl;
			return false;
		}

		// Transfer chunk data
		uint64_t count = 0;
		char* buf = current_gpu_ptr;

		while (count < current_chunk_size) {
			uint64_t bytes = current_chunk_size - count;

			ssize_t result = write(fd, buf, bytes);
			if (result != static_cast<ssize_t>(bytes)) {
				std::cerr << "Failed to transfer GPU chunk " << (chunk + 1)
						  << " to FPGA: " << strerror(errno) << std::endl;
				return false;
			}

			count += bytes;
			buf += bytes;
		}
	}

	return true;
}

/**
 * Transfer data from FPGA memory to GPU memory with chunking (low-level implementation)
 * Handles large transfers by breaking them into manageable chunks
 * @param fname Channel name for logging
 * @param fd File descriptor for DMA channel
 * @param gpu_buff GPU buffer pointer
 * @param size Total transfer size in bytes
 * @param fpga_offset FPGA memory offset
 * @return true on success, false on failure
 */
bool dma_manager::fpga_to_gpu(const char* fname, int fd, char* gpu_buff, uint64_t size, uint64_t fpga_offset){
	if(!initialized){
		std::cerr << "FPGA-GPU transfer not initialized" << std::endl;
		return false;
	}

	// Switch to GPU direct mode
	if(!set_gpu_mode()){
		std::cerr << "Failed to set GPU mode" << std::endl;
		return false;
	}

	// Calculate chunk parameters for large transfers
	uint64_t chunk_size_bytes = CHUNK_SIZE * sizeof(float);
	uint64_t total_chunks = (size + chunk_size_bytes - 1) / chunk_size_bytes;

	// Transfer data in chunks
	for (uint64_t chunk = 0; chunk < total_chunks; chunk++) {
		uint64_t current_offset 		= fpga_offset + (chunk * chunk_size_bytes);
		uint64_t current_chunk_size = std::min(chunk_size_bytes, size - (chunk * chunk_size_bytes));
		char* current_gpu_ptr 			= gpu_buff + (chunk * chunk_size_bytes);

		// Seek to FPGA offset for this chunk
		if (lseek64(fd, current_offset, SEEK_SET) != static_cast<off64_t>(current_offset)) {
			std::cerr << "Failed to seek to chunk offset " << current_offset << std::endl;
			return false;
		}

		// Transfer chunk data
		uint64_t count = 0;
		char* buf = current_gpu_ptr;

		while (count < current_chunk_size) {
			uint64_t bytes = current_chunk_size - count;

			ssize_t result = read(fd, buf, bytes);
			if (result != static_cast<ssize_t>(bytes)) {
				std::cerr << "Failed to transfer FPGA chunk " << (chunk + 1)
						  << " to GPU: " << strerror(errno) << std::endl;
				return false;
			}

			count += bytes;
			buf += bytes;
		}
	}

	return true;
}
/**
 * Initialize AXI-Lite register access by memory mapping BAR0
 * Maps the FPGA register space for direct register access
 * @return true on success, false on failure
 */
bool dma_manager::initialize_register(){
	if(reg_base_addr != nullptr){
		std::cout << "Register already initialized" << std::endl;
		return true;
	}

	// Open XDMA user device for register access
	int reg_fd = open("/dev/xdma0_user", O_RDWR);
	if(reg_fd < 0){
		std::cerr << "Failed to open /dev/xdma0_user: " << strerror(errno) << std::endl;
		return false;
	}

	// BAR0 maps to FPGA AXI-Lite register space starting at offset 0
	off_t bar_offset = 0x0;

	// Map 4KB register space
	reg_base_addr = mmap(nullptr, REG_SIZE, PROT_READ | PROT_WRITE,
											 MAP_SHARED, reg_fd, bar_offset);
	if(reg_base_addr == MAP_FAILED){
		std::cerr << "Failed to mmap register space: " << strerror(errno) << std::endl;
		close(reg_fd);
		reg_base_addr = nullptr;
		return false;
	}

	close(reg_fd);
	std::cout << "AXI-Lite register space mapped successfully" << std::endl;
	return true;
}
/**
 * Write 32-bit value to register at specified offset
 * @param base Base address of register space
 * @param offset Register offset in bytes
 * @param data Data to write
 */
inline void dma_manager::write_reg(void* base, uint32_t offset, uint32_t data){
	*((volatile uint32_t*)((char*)base + offset)) = data;
}

/**
 * Write to register at index i (0-based, 4-byte aligned)
 * @param i Register index
 * @param val Value to write
 */
void dma_manager::write_reg_i(int i, uint32_t val){
	write_reg(reg_base_addr, 0x00 + 4 * i, val);
}

/**
 * Read 32-bit value from register at specified offset
 * @param base Base address of register space
 * @param offset Register offset in bytes
 * @return Register value
 */
inline uint32_t dma_manager::read_reg(void* base, uint32_t offset){
	return *((volatile uint32_t*)((char*)base + offset));
}

/**
 * Read from register at index i (0-based, 4-byte aligned)
 * @param i Register index
 * @return Register value
 */
uint32_t dma_manager::read_reg_i(int i){
	return read_reg(reg_base_addr, 0x00 + 4 * i);
}

/**
 * Send command to FPGA with parameters
 * Writes command and data to registers 1, 2, 3, then triggers with CMD_DATA to register 0
 * @param command Command code
 * @param data0 First parameter
 * @param data1 Second parameter
 */
void dma_manager::host_setcommand(uint32_t command, uint32_t data0, uint32_t data1){
	write_reg_i(1, command);
	write_reg_i(2, data0);
	write_reg_i(3, data1);
	write_reg_i(0, CMD_DATA);
}


/**
 * Clean up all resources and reset state
 * Closes file descriptors, frees memory, and unmaps register space
 */
void dma_manager::cleanup(){
	// Close DMA channel file descriptors
	if(fd_h2c >= 0){
		close(fd_h2c);
		fd_h2c = -1;
	}
	if(fd_c2h >= 0){
		close(fd_c2h);
		fd_c2h = -1;
	}

	// Free host buffer memory
	if(host_buffer){
		delete[] host_buffer;
		host_buffer = nullptr;
	}

	// Unmap register space
	if(reg_base_addr != nullptr){
		munmap(reg_base_addr, REG_SIZE);
		reg_base_addr = nullptr;
	}

	// Reset state
	buffer_size = 0;
	initialized = false;
}

/*
 * C wrapper functions for Python bindings
 * Provides C interface to dma_manager class methods
 */

extern "C" {
	/**
	 * Create new DMA manager instance
	 * @return Pointer to dma_manager object
	 */
	void* dma_new(){
		return new dma_manager();
	}

	/**
	 * Delete DMA manager instance
	 * @param ptr Pointer to dma_manager object
	 */
	void dma_delete(void* ptr){
		delete static_cast<dma_manager*>(ptr);
	}

	/**
	 * Initialize DMA manager
	 * @param ptr DMA manager pointer
	 * @param table_size_bytes Buffer size in bytes
	 * @return true on success
	 */
	bool dma_initialize(void* ptr, size_t table_size_bytes){
		return static_cast<dma_manager*>(ptr)->initialize(table_size_bytes);
	}

	/**
	 * Transfer data from host to FPGA
	 * @param ptr DMA manager pointer
	 * @param data Float array pointer
	 * @param size Transfer size in bytes
	 * @param fpga_offset FPGA memory offset
	 * @return true on success
	 */
	bool dma_host_to_fpga(void* ptr, float* data, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->host_to_fpga(
			reinterpret_cast<const char*>(data), size, fpga_offset);
	}

	/**
	 * Transfer data from FPGA to host
	 * @param ptr DMA manager pointer
	 * @param data Float array pointer
	 * @param size Transfer size in bytes
	 * @param fpga_offset FPGA memory offset
	 * @return true on success
	 */
	bool dma_fpga_to_host(void* ptr, float* data, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->fpga_to_host(
			reinterpret_cast<char*>(data), size, fpga_offset);
	}

	/**
	 * Transfer data from host to FPGA (with custom file descriptor)
	 * @param ptr DMA manager pointer
	 * @param fname Channel name
	 * @param fd File descriptor
	 * @param data Float array pointer
	 * @param size Transfer size in bytes
	 * @param fpga_offset FPGA memory offset
	 * @return true on success
	 */
	bool dma_host_to_fpga_fd(void* ptr, const char* fname, int fd, float* data, uint64_t size, uint64_t fpga_offset) {
		return static_cast<dma_manager*>(ptr)->host_to_fpga(
			fname, fd, reinterpret_cast<const char*>(data), size, fpga_offset);
	}

	/**
	 * Transfer data from FPGA to host (with custom file descriptor)
	 * @param ptr DMA manager pointer
	 * @param fname Channel name
	 * @param fd File descriptor
	 * @param data Float array pointer
	 * @param size Transfer size in bytes
	 * @param fpga_offset FPGA memory offset
	 * @return true on success
	 */
	bool dma_fpga_to_host_fd(void* ptr, const char* fname, int fd, float* data, uint64_t size, uint64_t fpga_offset) {
		return static_cast<dma_manager*>(ptr)->fpga_to_host(
			fname, fd, reinterpret_cast<char*>(data), size, fpga_offset);
	}

	/**
	 * Set DMA mode to GPU direct
	 * @param ptr DMA manager pointer
	 * @return true on success
	 */
	bool dma_set_gpu_mode(void* ptr){
		return static_cast<dma_manager*>(ptr)->set_gpu_mode();
	}

	/**
	 * Set DMA mode to host
	 * @param ptr DMA manager pointer
	 * @return true on success
	 */
	bool dma_set_host_mode(void* ptr){
		return static_cast<dma_manager*>(ptr)->set_host_mode();
	}

	/**
	 * Transfer data from GPU to FPGA
	 * @param ptr DMA manager pointer
	 * @param gpu_ptr GPU buffer pointer
	 * @param size Transfer size in bytes
	 * @param fpga_offset FPGA memory offset
	 * @return true on success
	 */
	bool dma_gpu_to_fpga(void* ptr, void* gpu_ptr, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->gpu_to_fpga(reinterpret_cast<char*>(gpu_ptr), size, fpga_offset);
	}

	/**
	 * Transfer data from FPGA to GPU
	 * @param ptr DMA manager pointer
	 * @param gpu_ptr GPU buffer pointer
	 * @param size Transfer size in bytes
	 * @param fpga_offset FPGA memory offset
	 * @return true on success
	 */
	bool dma_fpga_to_gpu(void* ptr, void* gpu_ptr, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->fpga_to_gpu(reinterpret_cast<char*>(gpu_ptr), size, fpga_offset);
	}

	/**
	 * Transfer data from GPU to FPGA (with custom file descriptor)
	 * @param ptr DMA manager pointer
	 * @param fname Channel name
	 * @param fd File descriptor
	 * @param gpu_ptr GPU buffer pointer
	 * @param size Transfer size in bytes
	 * @param fpga_offset FPGA memory offset
	 * @return true on success
	 */
	bool dma_gpu_to_fpga_fd(void* ptr, const char* fname, int fd, void* gpu_ptr, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->gpu_to_fpga(fname, fd, reinterpret_cast<char*>(gpu_ptr), size, fpga_offset);
	}

	/**
	 * Transfer data from FPGA to GPU (with custom file descriptor)
	 * @param ptr DMA manager pointer
	 * @param fname Channel name
	 * @param fd File descriptor
	 * @param gpu_ptr GPU buffer pointer
	 * @param size Transfer size in bytes
	 * @param fpga_offset FPGA memory offset
	 * @return true on success
	 */
	bool dma_fpga_to_gpu_fd(void* ptr, const char* fname, int fd, void* gpu_ptr, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->fpga_to_gpu(fname, fd, reinterpret_cast<char*>(gpu_ptr), size, fpga_offset);
	}

	/**
	 * Check if DMA manager is initialized
	 * @param ptr DMA manager pointer
	 * @return true if initialized
	 */
	bool dma_is_initialized(void* ptr){
		return static_cast<dma_manager*>(ptr)->is_initialized();
	}

	/**
	 * Initialize register access
	 * @param ptr DMA manager pointer
	 * @return true on success
	 */
	bool dma_initialize_register(void* ptr){
		return static_cast<dma_manager*>(ptr)->initialize_register();
	}

	/**
	 * Send command to FPGA
	 * @param ptr DMA manager pointer
	 * @param command Command code
	 * @param data0 Parameter 0
	 * @param data1 Parameter 1
	 */
	void dma_host_setcommand(void* ptr, uint32_t command, uint32_t data0, uint32_t data1){
		static_cast<dma_manager*>(ptr)->host_setcommand(command, data0, data1);
	}

	/**
	 * Write to register at index
	 * @param ptr DMA manager pointer
	 * @param i Register index
	 * @param val Value to write
	 */
	void dma_write_reg_i(void* ptr, int i, uint32_t val){
		static_cast<dma_manager*>(ptr)->write_reg_i(i, val);
	}

	/**
	 * Read from register at index
	 * @param ptr DMA manager pointer
	 * @param i Register index
	 * @return Register value
	 */
	uint32_t dma_read_reg_i(void* ptr, int i){
		return static_cast<dma_manager*>(ptr)->read_reg_i(i);
	}

	/**
	 * Clean up DMA manager resources
	 * @param ptr DMA manager pointer
	 * @return Always returns 0
	 */
	bool dma_cleanup(void* ptr){
		static_cast<dma_manager*>(ptr)->cleanup();
		return 0;
	}
}