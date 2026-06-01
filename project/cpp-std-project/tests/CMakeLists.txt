# 单元测试可执行文件
set(TEST_SRCS
    test_calc.cpp
    test_str.cpp
)

add_executable(all_tests ${TEST_SRCS})

# 链接：业务库 + GTest
target_link_libraries(all_tests
    PRIVATE
        core_lib
        gtest
        gtest_main
)

# 注册测试
add_test(NAME AllTests COMMAND all_tests)