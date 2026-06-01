#include <gtest/gtest.h>
#include "utils/str.h"

TEST(StrTest, ToUpper)
{
    EXPECT_EQ(utils::to_upper("hello"), "HELLO");
    EXPECT_EQ(utils::to_upper("Cpp"), "CPP");
}