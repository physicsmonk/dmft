find_package(MPI REQUIRED)

# Add dependent library, this will build pugixml under build/ subdirectory in the project's build tree
add_subdirectory(../dependencies/pugixml dependencies/pugixml)

add_library(dmft
    bare_hamiltonian.cpp
    green_function.cpp
    ct_aux_imp_solver.cpp
    self_consistency.cpp)

target_include_directories(dmft PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    /home1/08105/tg874048/eigen-3.4.0  # Add Eigen include dir
    /work2/08105/tg874048/stampede2/apps/indicators/include  # Add indicator (header-only) include dir
    )

target_link_libraries(dmft PUBLIC
    MPI::MPI_CXX
    pugixml)
