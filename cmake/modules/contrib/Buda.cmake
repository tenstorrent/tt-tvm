
if(USE_BUDA_CODEGEN STREQUAL "ON")
  add_definitions(-DUSE_JSON_RUNTIME=1)
  tvm_file_glob(GLOB Buda_RELAY_CONTRIB_SRC src/relay/backend/contrib/buda/*.cc)
  list(APPEND COMPILER_SRCS ${Buda_RELAY_CONTRIB_SRC})
  list(APPEND COMPILER_SRCS ${JSON_RELAY_CONTRIB_SRC})

  #find_library(EXTERN_LIBRARY_Buda buda)
  #list(APPEND TVM_RUNTIME_LINKER_LIBS ${EXTERN_LIBRARY_Buda})
  #tvm_file_glob(GLOB Buda_CONTRIB_SRC src/runtime/contrib/buda/buda_json_runtime.cc)
  #list(APPEND RUNTIME_SRCS ${Buda_CONTRIB_SRC})
  message(STATUS "Build with Buda JSON runtime: " ${EXTERN_LIBRARY_Buda})
endif()


