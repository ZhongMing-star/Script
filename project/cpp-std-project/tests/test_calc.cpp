#include <gtest/gtest.h>
#include "math/calc.h"

TEST(CalcTest, AddTest)
{
    // 宏前缀	 失败行为	            适用场景
    // EXPECT_	报错但继续运行后续代码	 多个结果都想校验，看全部错误
    // ASSERT_	报错立刻终止当前用例     前置条件失败，后面代码无意义
    EXPECT_EQ(math::add(1, 2), 3);
    ASSERT_NE(math::add(5, 5), 9);
}

TEST(CalcTest, SubTest)
{
    EXPECT_EQ(math::sub(10, 3), 7);
}