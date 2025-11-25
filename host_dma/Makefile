CXX = g++
CXXFLAGS = -std=c++17 -O2 -fPIC
INCLUDES = -I./include
# LIBS = -lcuda  # CUDA 라이브러리 제거 (현재 사용하지 않음)

SOURCES = src/cpp/dma_manager.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = libdma_manager.so

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) -shared -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(TARGET)

.PHONY: all clean