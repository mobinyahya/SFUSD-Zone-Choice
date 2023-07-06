cmake_minimum_required(VERSION 3.22)
project(external C)

set(CMAKE_C_STANDARD 11)

add_executable(external
        external.c
        external_int.c)
