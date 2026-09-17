add_executable(caustic_3d caustic_3d.cpp)
target_link_libraries(caustic_3d imageplane raytracer ${cfitsio_libs})
