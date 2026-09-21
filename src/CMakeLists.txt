include_directories("raytracer")
add_subdirectory (raytracer)

include_directories("source_tracer")
add_subdirectory (source_tracer)

include_directories("mapper")
add_subdirectory (mapper)

include_directories("outflow")
add_subdirectory (outflow)

include_directories("ray_paths")
add_subdirectory (ray_paths)

include_directories("emissivity")
add_subdirectory (emissivity)

include_directories("imageplane")
add_subdirectory (imageplane)

include_directories("caustic")
add_subdirectory (caustic)

include_directories("reverb")
add_subdirectory (reverb)

include_directories("ray_transfer")
add_subdirectory (ray_transfer)

add_subdirectory (tests)