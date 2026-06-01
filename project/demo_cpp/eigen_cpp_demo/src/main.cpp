#include <iostream>
#include <Eigen/Dense>
using namespace std;

int main()
{
    std::cout << "Hello, Eigen!" << std::endl;

    std::cout << "======== 四则运算  =======" << std::endl;
    Eigen::MatrixXd A(2, 2), B(2, 2), C;
    A << 1, 2, 3, 4;
    B << 5, 6, 7, 8;

    std::cout << "Matrix A:\n" << A << std::endl;
    std::cout << "Matrix B:\n" << B << std::endl;

    C = A + B;                 // 矩阵加法
    std::cout << "A + B =\n" << C << std::endl;
    C = A - B;                 // 矩阵减法
    std::cout << "A - B =\n" << C << std::endl;
    C = A * B;                 // 矩阵乘法
    std::cout << "A * B =\n" << C << std::endl; 
    C = A.array() * B.array(); // 元素级乘法（Hadamard积）
    std::cout << "A.array() * B.array() =\n" << C << std::endl;
    C = A / 2.0;               // 标量除法
    std::cout << "A / 2.0 =\n" << C << std::endl;
    C = A.array() / B.array(); // 元素级除法
    std::cout << "A.array() / B.array() =\n" << C << std::endl;

    return 0;
}