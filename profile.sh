target=./build/gemm_fp32_app

ncu -f -o gemm_fp32_app --import-source yes \
    --set full \
    --kernel-id ::hopper_gemm_fp32_kernel:2 \
    $target