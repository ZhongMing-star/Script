#include "utils/str.h"
#include <cctype>

namespace utils {

std::string to_upper(std::string s)
{
    for (char& c : s)
    {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

} // namespace utils