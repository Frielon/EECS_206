find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)

rosidl_generate_interface(${PROJECT_NAME}
    "msg/mydata.msg
    DEPENDENCIES builtin_interfaces
)