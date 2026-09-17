<<<<<<< HEAD
add_executable(caustic_3d caustic_3d.cpp)
target_link_libraries(caustic_3d imageplane raytracer ${cfitsio_libs})
=======
add_executable(caustic_discplane caustic_discplane.cpp)
target_link_libraries(caustic_discplane imageplane raytracer ${cfitsio_libs})

add_executable(caustic_sourceplane caustic_sourceplane.cpp)
target_link_libraries(caustic_sourceplane imageplane raytracer ${cfitsio_libs})

add_executable(caustic_plane caustic_plane.cpp)
target_link_libraries(caustic_plane imageplane raytracer ${cfitsio_libs})
>>>>>>> 4ccb635e31ef642a9fede1592507aefdc5b218ae
