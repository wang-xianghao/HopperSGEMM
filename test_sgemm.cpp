#include "hopper_sgemm.hpp"
#include "hopper_sgemm_utils.hpp"

int main()
{
    const int m{4096};
    const int n{4096};
    const int k{4096};
    const float alpha{0.5f};
    const float beta{0.5f};

    hopper_sgemm(0, 0, 0, 0, nullptr, 0, nullptr, 0, 0, nullptr, 0);

    return 0;
}