cmake_minimum_required(VERSION 3.10)
project(DAO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# [SLLM NEW STRATEGY] Manual configuration for LibTorch to bypass faulty scripts.
# Set this variable to the root of your unzipped libtorch directory.
set(LIBTORCH_ROOT /replace_path/Libs/CPP/libtorch)

# Manually add the include directories for LibTorch headers.
include_directories(${LIBTORCH_ROOT}/include)
include_directories(${LIBTORCH_ROOT}/include/torch/csrc/api/include)

# Manually add the library directory for the linker to find .so files.
link_directories(${LIBTORCH_ROOT}/lib)

# --- Find CUDA and conditionally compile ---
# This will find the CUDA toolkit on your system.
find_package(CUDA)
if(CUDA_FOUND)
    message(STATUS "CUDA Toolkit found. Configuring for GPU support.")
    # Add CUDA as a language to the project
    enable_language(CUDA)
    # This flag is often needed for PyTorch CUDA projects
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fPIC")
else()
    message(WARNING "CUDA Toolkit not found. Building in CPU-only mode.")
endif()


find_package(OpenMP REQUIRED)

# --- Manual SentencePiece linking ---
add_library(sentencepiece SHARED IMPORTED GLOBAL)
set_target_properties(sentencepiece PROPERTIES
    IMPORTED_LOCATION "/usr/local/lib/libsentencepiece.so"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include"
)

add_library(sentencepiece_train SHARED IMPORTED GLOBAL)
set_target_properties(sentencepiece_train PROPERTIES
    IMPORTED_LOCATION "/usr/local/lib/libsentencepiece_train.so"
    INTERFACE_INCLUDE_DIRECTORIES "/usr/local/include"
)


# --- Main DAO library ---
# Note: We are now building trainer.cpp directly into the 'chat' executable
# to simplify the linking process.
add_library(dao_core
    src/rdse.cpp
    src/grid_cell_encoder.cpp
    src/text_sdr_encoder.cpp
    src/spatial_pooler.cpp
    src/resonance_layer.cpp
    src/temporal_memory.cpp
    src/attention.cpp
    src/conversational_generator.cpp
    src/dao_model.cpp
    src/trainer.cpp
)

target_include_directories(dao_core PUBLIC
    src
    /replace_path/Libs/CPP/eigen-3.4.0
    /replace_path/Libs/CPP/cereal-1.3.2/include
)


# --- Executable to train tokenizer ---
add_executable(train_tokenizer src/train_tokenizer.cpp)
target_link_libraries(train_tokenizer PRIVATE dao_core sentencepiece_train sentencepiece)


# --- Main Chat Executable ---
# We now include trainer.cpp here to ensure it has access to the Torch headers.
add_executable(chat src/chat.cpp src/trainer.cpp)

target_compile_definitions(chat PRIVATE
    ENABLE_PROFILING
    ENABLE_TRAINING_PROFILING
)

# --- [SLLM FINAL LINKING w/ LINKER FLAGS] ---
# Conditionally link CUDA libraries if they were found.
if(CUDA_FOUND)
    message(STATUS "Linking chat executable against CUDA libraries.")
    target_link_libraries(chat PRIVATE
        dao_core
        sentencepiece
        OpenMP::OpenMP_CXX
        
        # Force the linker to keep a reference to libtorch.so, which is
        # necessary for the dynamic loading of CUDA components at runtime.
        "-Wl,--no-as-needed"
        torch
        "-Wl,--as-needed"

        # Link the rest of the libraries
        torch_cuda
        c10_cuda
        torch_cpu
        c10
        ${CUDA_CUDART_LIBRARY}
    )
else()
    message(STATUS "Linking chat executable against CPU-only libraries.")
    target_link_libraries(chat PRIVATE
        dao_core
        sentencepiece
        OpenMP::OpenMP_CXX
        torch
        torch_cpu
        c10
    )
endif()


message(STATUS "DAO project configured with 'chat' and 'train_tokenizer' executables.")