set(VECTORIZATION "" CACHE STRING "Specify vectorization")
#if("${VECTORIZATION}" STREQUAL "AVX512_KNL")
#    set(VECTOR_FLAG "-xMIC-AVX512")
#elseif("${VECTORIZATION}" STREQUAL "AVX512_SKX")
#    set(VECTOR_FLAG "-xCORE-AVX512")
#elseif("${VECTORIZATION}" STREQUAL "AVX2")
#    set(VECTOR_FLAG "-march=core-avx2")
#else()
#    set(VECTOR_FLAG "")
#endif()
set(FPMODEL "" CACHE STRING "Specify math precision")

add_library(pugixml pugixml.cpp)

target_include_directories(pugixml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(pugixml PUBLIC -std=c++17 "${FPMODEL}" "${VECTORIZATION}")  # $<$<CONFIG:RELEASE>:-xMIC-AVX512> "${VECTOR_FLAG}" -fp-model precise
target_link_options(pugixml PUBLIC -std=c++17 "${FPMODEL}" "${VECTORIZATION}")  #  $<$<CONFIG:RELEASE>:-xMIC-AVX512> "${VECTOR_FLAG}" -fp-model precise
