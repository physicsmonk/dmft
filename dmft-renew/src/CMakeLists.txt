find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(GMP REQUIRED)
find_package(MPFR REQUIRED)

# Add dependent library, this will build pugixml under build/ subdirectory in the project's build tree
add_subdirectory(../dependencies/pugixml dependencies/pugixml)

add_library(dmft
    bare_hamiltonian.cpp
    green_function.cpp
    ct_aux_imp_solver.cpp
    self_consistency.cpp)

target_include_directories(dmft PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../dependencies
    )

target_link_libraries(dmft PUBLIC
    Eigen3::Eigen
    ${MPFR_LIBRARIES}
    ${GMP_LIBRARIES}
    MPI::MPI_CXX
    pugixml
    )
