
if(NOT DEFINED PKG_PATH)
    message(FATAL_ERROR "PKG_PATH not set")
endif()

set(OP_HEADER_PATH ${PKG_PATH}/tools/codegen/include)
set(HEADER_PATH ${PKG_PATH}/runtime)

message(STATUS "operator lib path: ${OP_LIB}")
message(STATUS "operator header path: ${OP_HEADER_PATH}")

add_compile_definitions(NOT_USE_STL)

include_directories(${OP_HEADER_PATH})
include_directories(${HEADER_PATH})
include_directories(${HEADER_PATH}/include)
include_directories(${MS_ROOT_DIR})
include_directories(${MS_ROOT_DIR}/include)

include(net.cmake)
function(create_library)
    if(OP_SRC)
        add_custom_command(TARGET net
                POST_BUILD
                COMMAND rm -rf tmp
                COMMAND mkdir tmp
                COMMAND cd tmp && ar -x ${OP_LIB}
                COMMAND cd tmp && ar -x ${WRAPPER_LIB}
                COMMAND echo "raw static library ${library_name} size:"
                COMMAND ls -lh ${library_name}
                COMMAND mv ${library_name} ./tmp && cd tmp && ar -x ${library_name}
                COMMENT "unzip raw static library ${library_name}"
                )

        foreach(object_file ${OP_SRC})
            add_custom_command(TARGET net POST_BUILD COMMAND mv ./tmp/${object_file} .)
        endforeach()
        add_custom_command(TARGET net
                POST_BUILD
                COMMAND ar cr ${library_name} *.o
                COMMAND ranlib ${library_name}
                COMMAND echo "new static library ${library_name} size:"
                COMMAND ls -lh ${library_name}
                COMMAND rm -rf tmp && rm -rf *.o
                COMMENT "generate specified static library ${library_name}"
                )
    else()
        add_custom_command(TARGET net
                POST_BUILD
                COMMAND rm -f ${library_name}
                COMMAND ar cr ${library_name}
                COMMAND ranlib ${library_name}
                COMMAND echo "generate empty static library ${library_name}"
                COMMAND ls -lh ${library_name}
                COMMENT "generate empty static library ${library_name}"
                )
    endif()
endfunction(create_library)
string(CONCAT library_name "lib" net ".a")
create_library()

