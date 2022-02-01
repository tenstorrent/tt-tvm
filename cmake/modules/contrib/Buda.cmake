
if(USE_BUDA_CODEGEN STREQUAL "ON")
  add_definitions(-DUSE_JSON_RUNTIME=1)
  tvm_file_glob(GLOB Buda_RELAY_CONTRIB_SRC src/relay/backend/contrib/buda/*.cc)
  list(APPEND COMPILER_SRCS ${Buda_RELAY_CONTRIB_SRC})

  SET(BUDA_LIB_PATH ${CMAKE_SOURCE_DIR}/../../build/lib)
  find_library(EXTERN_LIBRARY_CORE pybuda_csrc PATHS ${BUDA_LIB_PATH} NO_DEFAULT_PATH) 
  message(STATUS "EXTERN_LIBRARY_CORE: " ${EXTERN_LIBRARY_CORE})

  list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_CORE})

  #SET(Torch_DIR ${CMAKE_SOURCE_DIR}/../libtorch/share/cmake/Torch)
  #message(STATUS "Torch_DIR: " ${Torch_DIR})
  #find_package(Torch REQUIRED)
  #list(APPEND TVM_RUNTIME_LINKER_LIBS ${TORCH_LIBRARIES})
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  #message(STATUS "TORCH_CXX_FLAGS: " ${TORCH_CXX_FLAGS})
  #message(STATUS "TORCH_LIBRARIES: " ${TORCH_LIBRARIES})
#
  #SET(TORCH_PYTHON_PATH ${CMAKE_SOURCE_DIR}/../libtorch/lib)
  #find_library(TORCH_PYTHON torch_python PATHS ${TORCH_PYTHON_PATH})
  #message(STATUS "TORCH_PYTHON: " ${TORCH_PYTHON})
  #list(APPEND TVM_RUNTIME_LINKER_LIBS ${TORCH_PYTHON})

  #message(STATUS "TORCH_INCLUDE_DIRS: " ${TORCH_INCLUDE_DIRS})
  #add_subdirectory(${CMAKE_SOURCE_DIR}/../pybind11 ${CMAKE_SOURCE_DIR}/../pybind11/build)

  find_package (Python3 COMPONENTS Interpreter Development)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${Python3_LIBRARIES})

  SET(BUDA_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/../..)
  list(APPEND BUDA_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/../../pybuda/csrc/)
  list(APPEND BUDA_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/../../model)
  list(APPEND BUDA_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/../../placer)
  list(APPEND BUDA_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/../../third_party/fmt)
  list(APPEND BUDA_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/../../third_party/json)
  list(APPEND BUDA_INCLUDE_DIRS ${CMAKE_SOURCE_DIR}/../../build/python_env/lib/python3.8/site-packages/pybind11/include)
  list(APPEND BUDA_INCLUDE_DIRS /usr/include/python3.8)

  include_directories(SYSTEM ${BUDA_INCLUDE_DIRS})
  tvm_file_glob(GLOB Buda_CONTRIB_SRC src/runtime/contrib/buda/buda_json_runtime.cc)
  list(APPEND RUNTIME_SRCS ${Buda_CONTRIB_SRC})
endif()


