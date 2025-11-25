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

dma_manager::dma_manager()
    : fd_h2c(-1), fd_c2h(-1), host_buffer(nullptr), 
      buffer_size(0), initialized(false), reg_base_addr(nullptr) {  // 추가
}
dma_manager::~dma_manager() {
    cleanup();
}

bool dma_manager::initialize(size_t table_size_bytes){
	try{
		fd_h2c = open(DEVICE_NAME_H2C, O_RDWR);
		if(fd_h2c<0){
			std::cerr << "Failed to open H2C devices: "<< strerror(errno) << std::endl;
			return false;
		}
		fd_c2h = open(DEVICE_NAME_C2H, O_RDWR);
		if(fd_c2h<0){
			std::cerr << "Failed to open C2H devices: "<< strerror(errno) << std::endl;
			close(fd_h2c); // close fd_h2c channel, which is already open
			return false;
		}
		// allocate host memory
		host_buffer = new float[table_size_bytes / sizeof(float)];
		if(!host_buffer){
			std::cerr << "Failed to allocate host memory" << std::endl;
			cleanup();
			return false;
		}
		buffer_size = table_size_bytes;
		initialized = true;
		
		std::cout << "😀KB Transfer operation initialized: "
							<< (table_size_bytes/ (1024*1024)) << " [MB]" << std::endl;
		return true;
	} catch(const std::exception& e){
		std::cerr << "🥶KV Transfer initialization failed: " << e.what() << std::endl;
		cleanup();
		return false;
	}
}

// 이 함수를 추가!
bool dma_manager::host_to_fpga(const char* fname, int fd, const char* host_buff, uint64_t size, uint64_t fpga_offset){
	if(!initialized){
		std::cerr << "KV Transfer not initialized" << std::endl;
		return false;
	}
	
	// set offset
	if(lseek(fd, fpga_offset, SEEK_SET)!= static_cast<off_t>(fpga_offset)){
		std::cerr << "Failed to seek to offset " << fpga_offset << std::endl;
		return false;
	}

	std::cout << "Transferring " << (static_cast<double>(size) /(1024*1024))<< "MB to FPGA" << std::endl;

	uint64_t count = 0;
	const char* buf = host_buff;

	while (count < size){
		std::cout<<"while LOOP started"<<std::endl;
		uint64_t bytes = size - count;
		if(bytes > RW_MAX_SIZE){
			bytes = RW_MAX_SIZE;
		}
		std::cout<<"start write operation"<<std::endl;
		ssize_t result = write(fd, buf, bytes);
		if(result == static_cast<ssize_t>(bytes)){
			// OK
		}	else{	
			std::cerr << "🥶 Failed to transfer to FPGA: "<<strerror(errno) << std::endl;
			return false;
		}
		count += bytes;
		buf += bytes;
		std::cout << "Writing "<<(count /(1024*1024))<<" / "<<(size / (1024*1024))<<" [MB]"<<std::endl;
	}
	
	if(count == static_cast<ssize_t>(size)){
		std::cout << "😀 Successfully transferred to FPGA!" << std::endl;
		return true;
	} else{
		std::cerr << "🥶 Failed to transfer to FPGA: "<<strerror(errno) << std::endl;
		return false;
	}
}

bool dma_manager::host_to_fpga(const char* host_buff, uint64_t size, uint64_t fpga_offset){
	return host_to_fpga("H2C", fd_h2c, host_buff, size, fpga_offset);
}

bool dma_manager::fpga_to_host(char* host_buff, uint64_t size, uint64_t fpga_offset){
	return fpga_to_host("C2H", fd_c2h, host_buff, size, fpga_offset);
}

bool dma_manager::fpga_to_host(const char* fname, int fd, char* host_buff, uint64_t size, uint64_t fpga_offset){
	if(!initialized){
		std::cerr<<"KV Transfer not initialized"<<std::endl;
		return false;
	}
	
	if(lseek(fd, fpga_offset, SEEK_SET)!=static_cast<off_t>(fpga_offset)){
		std::cerr <<"Failed to seek to offset " << fpga_offset<< std::endl;
		return false;
	}
	std::cout << "Reading "<<(size/(1024*1024))<<" [MB] from FPGA"<< std::endl;
	
	uint64_t 	count 	= 0;
	char* 		buf 		= host_buff;
	uint64_t 	offset 	= fpga_offset;
	
	while(count < size){
		uint64_t bytes = size - count;
		if(bytes > RW_MAX_SIZE){
			bytes = RW_MAX_SIZE;
		}
		ssize_t result = read(fd, buf, bytes);		

		if(result > 0 && result >= 4) {  // ← 추가!
			uint32_t* data_ptr = reinterpret_cast<uint32_t*>(buf);
		}
		
		if(result == static_cast<ssize_t>(bytes)){
			// Success
		} else { 
			std::cerr << "🥶 Failed to read from FPGA: " << strerror(errno) << std::endl;
			std::cerr << "[DEBUG] result=" << result << ", bytes=" << bytes << std::endl;  // ← 추가!
			return false;
		}
		
		count 	+=	bytes;
		buf		+= 	bytes;
		offset 	+=	bytes;

		std::cout << "Read " << (count / (1024*1024)) << " / " << (size / (1024 * 1024)) << " [MB]"<<std::endl;
	}
	
	if(count == static_cast<ssize_t>(size)){
		std::cout << "😀 Successfully read from FPGA!" << std::endl;
		
		// 최종 확인
		uint32_t* final_data = reinterpret_cast<uint32_t*>(host_buff);  // ← 추가!
		std::cout << "[DEBUG] Final buffer first word: 0x" << std::hex << final_data[0] << std::dec << std::endl;
		
		return true;
	}	else{
		std::cerr << "🥶 Failed to read from FPGA: "<<strerror(errno) << std::endl;
		return false;
	}
}
void dma_manager::set_device_mode(int fd, int ioctl_cmd, const char* mode_name) {
    if (ioctl(fd, ioctl_cmd) < 0) {
        throw std::runtime_error(std::string("IOCTL Fail (") + mode_name + "): " + strerror(errno));
    }
}
bool dma_manager::set_host_mode(){
	try{
		set_device_mode(fd_h2c, IOCTL_XDMA_DIR_TO_HOST, "H2C Host");
		set_device_mode(fd_c2h, IOCTL_XDMA_DIR_TO_HOST, "C2H Host");
		return true;
	} catch(const std::exception& e){
		std::cerr <<"Host Mode Fail!: "<<e.what() <<std::endl;
		return false;
	}
}

bool dma_manager::set_gpu_mode(){
	try{
		set_device_mode(fd_h2c, IOCTL_XDMA_DIR_TO_GPU, "H2C GPU Direct");
		set_device_mode(fd_c2h, IOCTL_XDMA_DIR_TO_GPU, "C2H GPU_DIRECT");
		return true;
	} catch(const std::exception& e){
		std::cerr <<"GPU Mode Fail!: "<<e.what() <<std::endl;
		return false;
	}
}

bool dma_manager::gpu_to_fpga(char* gpu_buff, uint64_t size, uint64_t fpga_offset){
	return gpu_to_fpga("H2C-GPU", fd_h2c, gpu_buff, size, fpga_offset);
};
bool dma_manager::fpga_to_gpu(char* gpu_buff, uint64_t size, uint64_t fpga_offset){
	return fpga_to_gpu("C2H-GPU", fd_c2h, gpu_buff, size, fpga_offset);
};


// [write]
bool dma_manager::gpu_to_fpga(const char* fname, int fd, char* gpu_buff, uint64_t size, uint64_t fpga_offset){
	if(!initialized){ 
		std::cerr<<"GPU-FPGA Transfer not initialized"<<std::endl;
		return false;
	}
	// GPU direct Mode 전환
	if(!set_gpu_mode()){
		std::cerr <<"Failed to set GPU mode"<< std::endl;
		return false;
	}
	
	// Chunk 설정
	uint64_t chunk_size_bytes = CHUNK_SIZE * sizeof(float);
	uint64_t total_chunks = (size + chunk_size_bytes - 1) / chunk_size_bytes;
	
	std::cout << "분할 GPU->FPGA 전송 시작: 총 " << total_chunks << "개 청크 (" 
			  << CHUNK_SIZE_MB << "MB/청크)" << std::endl;
	std::cout << "총 크기: " << (static_cast<double>(size)/(1024*1024)) << " MB" << std::endl;
	
	for (uint64_t chunk = 0; chunk < total_chunks; chunk++) {
		uint64_t current_offset = fpga_offset + (chunk * chunk_size_bytes);
		uint64_t current_chunk_size = std::min(chunk_size_bytes, size - (chunk * chunk_size_bytes));
		char* current_gpu_ptr = gpu_buff + (chunk * chunk_size_bytes);
		
		// FPGA offset 설정
		if (lseek(fd, current_offset, SEEK_SET) != static_cast<off_t>(current_offset)) {
			std::cerr << "Failed to seek to chunk offset " << current_offset << std::endl;
			return false;
		}
		
		std::cout << "Write 청크 " << (chunk + 1) << "/" << total_chunks 
				  << " 전송 중... (" << (current_chunk_size / (1024*1024)) << "MB)" << std::endl;
		
		// 청크별 전송 (기존 while 루프 방식 사용)
		uint64_t count = 0;
		char* buf = current_gpu_ptr;
		
		while (count < current_chunk_size) {
			uint64_t bytes = current_chunk_size - count;
			if (bytes > RW_MAX_SIZE) {
				bytes = RW_MAX_SIZE;
			}
			
			ssize_t result = write(fd, buf, bytes);
			if (result != static_cast<ssize_t>(bytes)) {
				std::cerr << "Failed to transfer chunk " << (chunk + 1) 
						  << " (GPU->FPGA): " << strerror(errno) << std::endl;
				return false;
			}
			
			count += bytes;
			buf += bytes;
		}
		
		std::cout << "Write 청크 " << (chunk + 1) << " 전송 완료" << std::endl;
	}
	
	std::cout << "\n=== 분할 GPU->FPGA 전송 완료 ===" << std::endl;
	
	return true;
}

// [read]  
bool dma_manager::fpga_to_gpu(const char* fname, int fd, char* gpu_buff, uint64_t size, uint64_t fpga_offset){
	if(!initialized){
		std::cerr << "FPGA-GPU Transfer not initialized" << std::endl;
		return false;
	}
	// GPU Direct 모드 전환
	if(!set_gpu_mode()){
		std::cerr << "Failed to set GPU mode" << std::endl;
		return false;
	}
	
	// Chunk 설정
	uint64_t chunk_size_bytes = CHUNK_SIZE * sizeof(float);
	uint64_t total_chunks = (size + chunk_size_bytes - 1) / chunk_size_bytes;
	
	std::cout << "분할 FPGA->GPU 전송 시작: 총 " << total_chunks << "개 청크 (" 
			  << CHUNK_SIZE_MB << "MB/청크)" << std::endl;
	std::cout << "총 크기: " << (static_cast<double>(size)/(1024*1024)) << " MB" << std::endl;
	
	for (uint64_t chunk = 0; chunk < total_chunks; chunk++) {
		uint64_t current_offset 	= fpga_offset + (chunk * chunk_size_bytes);
		uint64_t current_chunk_size = std::min(chunk_size_bytes, size - (chunk * chunk_size_bytes));
		char* current_gpu_ptr 		= gpu_buff + (chunk * chunk_size_bytes);
		
		// FPGA offset 설정
		if (lseek(fd, current_offset, SEEK_SET) != static_cast<off_t>(current_offset)) {
			std::cerr << "Failed to seek to chunk offset " << current_offset << std::endl;
			return false;
		}
		
		std::cout << "Read 청크 " << (chunk + 1) << "/" << total_chunks 
				  << " 전송 중... (" << (current_chunk_size / (1024*1024)) << "MB)" << std::endl;
		std::cout << "청크 " << (chunk + 1) << " 오프셋: " << current_offset 
				  << " (0x" << std::hex << current_offset << std::dec << ")" << std::endl;
		
		// 청크별 전송 (기존 while 루프 방식 사용)
		uint64_t count 	= 0;
		char* buf 		= current_gpu_ptr;
		
		while (count < current_chunk_size) {
			uint64_t bytes = current_chunk_size - count;
			if (bytes > RW_MAX_SIZE) {
				bytes = RW_MAX_SIZE;
			}
			
			ssize_t result = read(fd, buf, bytes);
			if (result != static_cast<ssize_t>(bytes)) {
				std::cerr << "🥶 Failed to read chunk " << (chunk + 1) 
						  << " (FPGA->GPU): " << strerror(errno) 
						  << " at offset 0x" << std::hex << (current_offset + count) << std::dec << std::endl;
				return false;
			}
			
			count += bytes;
			buf += bytes;
		}
		
		std::cout << "Read 청크 " << (chunk + 1) << " 전송 완료" << std::endl;
	}
	
	std::cout << "\n=== 분할 FPGA->GPU 전송 완료 ===" << std::endl;
	
	return true;
}
bool dma_manager::initialize_register(){
	if(reg_base_addr != nullptr){
			std::cout << "Register already initialized" <<std::endl;
			return true;
	}
	int reg_fd = open("/dev/xdma0_user", O_RDWR);
	if(reg_fd < 0){
			std::cerr <<"Failed to open /dev/xdma0_user: " << strerror(errno) << std::endl;
			return false;
	}

	// BAR0 (user BAR)는 FPGA AXI-Lite 베이스 주소에 매핑됨
	off_t bar_offset = 0x0;

	// 4KB Register Space Mapped
	reg_base_addr = mmap(nullptr, REG_SIZE, PROT_READ | PROT_WRITE, 
											 MAP_SHARED, reg_fd, bar_offset);
	if(reg_base_addr == MAP_FAILED){
			std::cerr << "Failed to mmap register space: " << strerror(errno) << std::endl;
			close(reg_fd);
			reg_base_addr = nullptr;
			return false;
	}
	close(reg_fd);
	std::cout << "Register mapped to BAR successfully!" << std::endl;
	return true;
}
inline void dma_manager::write_reg(void* base, uint32_t offset, uint32_t data){
	*((volatile uint32_t*)((char*)base+offset)) = data;
}

void dma_manager::write_reg_i(int i, uint32_t val){
	write_reg(reg_base_addr, 0x00+4*i, val);
}

inline uint32_t dma_manager::read_reg(void* base, uint32_t offset){
	return *((volatile uint32_t*)((char*)base+offset));
}

uint32_t dma_manager::read_reg_i(int i){
	return read_reg(reg_base_addr, 0x00+4*i);
}

void dma_manager::host_setcommand(uint32_t command, uint32_t data0, uint32_t data1){
	write_reg_i(1, 	command);
	write_reg_i(2, 	data0);
	write_reg_i(3,	data1);
	write_reg_i(0, 	CMD_DATA);
}


void dma_manager::cleanup(){
	if(fd_h2c >= 0){
		close(fd_h2c);
		fd_h2c = -1;
	}
	if(fd_c2h >= 0){
		close(fd_c2h);
		fd_c2h = -1;
	}
	if(host_buffer){
		delete[] host_buffer;
		host_buffer = nullptr;
	}
	buffer_size = 0;
	initialized = false;

	if(reg_base_addr != nullptr){
		munmap(reg_base_addr, REG_SIZE);
		reg_base_addr = nullptr;
	}
}

	/* 'new' create object on the Heap Memory
		return the pointer which point that object

		'delete' delete the object on the Heap memory
		deallocate Memory space
	*/

extern "C" {
	void* dma_new(){
		return new dma_manager(); 
	} // fpga_kv_transfer() -> void * 로 반환된다.

	void dma_delete(void* ptr){
		delete static_cast<dma_manager*>(ptr);
	}

	bool dma_initialize(void* ptr, size_t table_size_bytes){
		return static_cast<dma_manager*>(ptr)->initialize(table_size_bytes);
	}

	bool dma_host_to_fpga(void* ptr, float* data, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->host_to_fpga(
			reinterpret_cast<const char*>(data), size, fpga_offset);
	}

	bool dma_fpga_to_host(void* ptr, float* data, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->fpga_to_host(
			reinterpret_cast<char*>(data), size, fpga_offset);
	}

    bool dma_host_to_fpga_fd(void* ptr, const char* fname, int fd, float* data, uint64_t size, uint64_t fpga_offset) {
        return static_cast<dma_manager*>(ptr)->host_to_fpga(
            fname, fd, reinterpret_cast<const char*>(data), size, fpga_offset);
    }
    
    bool dma_fpga_to_host_fd(void* ptr, const char* fname, int fd, float* data, uint64_t size, uint64_t fpga_offset) {
        return static_cast<dma_manager*>(ptr)->fpga_to_host(
            fname, fd, reinterpret_cast<char*>(data), size, fpga_offset);
    }
    
	bool dma_set_gpu_mode(void* ptr){
		return static_cast<dma_manager*>(ptr)->set_gpu_mode();
	}

	bool dma_set_host_mode(void* ptr){
		return static_cast<dma_manager*>(ptr)->set_host_mode();
	}

	bool dma_gpu_to_fpga(void* ptr, void* gpu_ptr, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->gpu_to_fpga(reinterpret_cast<char*>(gpu_ptr), size, fpga_offset);
	}

	bool dma_fpga_to_gpu(void* ptr, void* gpu_ptr, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->fpga_to_gpu(reinterpret_cast<char*>(gpu_ptr), size, fpga_offset);
	}
	
	bool dma_gpu_to_fpga_fd(void* ptr, const char* fname, int fd, void* gpu_ptr, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->gpu_to_fpga(fname, fd, reinterpret_cast<char*>(gpu_ptr), size, fpga_offset);
	}
	
	bool dma_fpga_to_gpu_fd(void* ptr, const char* fname, int fd, void* gpu_ptr, uint64_t size, uint64_t fpga_offset){
		return static_cast<dma_manager*>(ptr)->fpga_to_gpu(fname, fd, reinterpret_cast<char*>(gpu_ptr), size, fpga_offset);
	}

	bool dma_is_initialized(void* ptr){
		return static_cast<dma_manager*>(ptr)->is_initialized();
	}

	bool dma_initialize_register(void* ptr){
		return static_cast<dma_manager*>(ptr)->initialize_register();
	}

	void dma_host_setcommand(void* ptr, uint32_t command, uint32_t data0, uint32_t data1){
		static_cast<dma_manager*>(ptr)->host_setcommand(command, data0, data1);
	}

	void dma_write_reg_i(void* ptr, int i, uint32_t val){
		static_cast<dma_manager*>(ptr)->write_reg_i(i, val);
	}

	uint32_t dma_read_reg_i(void* ptr, int i){
		return static_cast<dma_manager*>(ptr)->read_reg_i(i);
	}

	bool dma_cleanup(void* ptr){
		static_cast<dma_manager*>(ptr)->cleanup();
		return 0;
	}
	
}