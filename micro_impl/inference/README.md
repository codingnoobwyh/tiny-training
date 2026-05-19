This directory is only for forward inference validation.

It exports the baseline model to ONNX, exports MNIST payloads, creates ONNX Runtime golden output, converts fp32/int8 MindSpore Lite Micro models, and runs benchmark cosine checks.

Use:

```bash
bash micro_impl/forward_inference/build.sh benchmark
bash micro_impl/forward_inference/build.sh benchmark quant
bash micro_impl/forward_inference/build.sh riscv
bash micro_impl/forward_inference/build.sh riscv quant
```

Generated files stay under `micro_impl/forward_inference/artifacts/`.

