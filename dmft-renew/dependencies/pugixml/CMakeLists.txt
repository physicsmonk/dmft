add_library(pugixml pugixml.cpp)

target_include_directories(pugixml PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_options(pugixml PUBLIC -std=c++17 $<$<CONFIG:RELEASE>:-xMIC-AVX512>)
target_link_options(pugixml PUBLIC -std=c++17 $<$<CONFIG:RELEASE>:-xMIC-AVX512>)
