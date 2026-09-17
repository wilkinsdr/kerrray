add_executable(disc_ent_line_rbin disc_ent_line_rbin.cpp)
target_link_libraries(disc_ent_line_rbin pointsource raytracer ${cfitsio_libs} ${HDF5_CXX_LIBRARIES})
target_include_directories(disc_ent_line_rbin PRIVATE ${HDF5_CXX_INCLUDE_DIRS})