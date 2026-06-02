
/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/model0/weight0.h"
#include "src/model0/net0.h"
#include <stdio.h>

#ifdef BENCHMARK

#include <stdio.h>

static void PrintTop10Grads(const char *name, const float *data, int length) {
    printf("   %s top10:", name);
    int count = length < 10 ? length : 10;
    for (int i = 0; i < count; ++i) {
        printf(" %.9g", data[i]);
    }
    printf("\n");
}

static void DumpGradsToFile(const char *name, const float *data, int length) {
    char path[128];
    snprintf(path, sizeof(path), "micro_grad_%s.txt", name);
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        printf("   dump %s failed\n", path);
        return;
    }
    for (int i = 0; i < length; ++i) {
        fprintf(file, "%.9g\n", data[i]);
    }
    fclose(file);
}

static void DumpInt8TensorToFile(const char *name, const int8_t *data,
                                 int length) {
    char path[128];
    snprintf(path, sizeof(path), "micro_activation_%s.txt", name);
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        printf("   dump %s failed\n", path);
        return;
    }
    for (int i = 0; i < length; ++i) {
        fprintf(file, "%d\n", (int)data[i]);
    }
    fclose(file);
}

static void PrintAndDumpGrads(const char *name, const float *data, int length) {
    PrintTop10Grads(name, data, length);
    DumpGradsToFile(name, data, length);
}

#else

#define PrintAndDumpGrads(name, data, length) ((void)0)

#endif

static void FullyConnectedQASBackward(const float *dy,
                                      const int8_t *input, const int8_t *weight,
                                      const float *effective_scale,
                                      int input_zp, int input_size,
                                      int output_size, float dy_scale,
                                      float *dx, float *dw,
                                      float *db) {
    for (int o = 0; o < output_size; ++o) {
        float dlinear_out = dy[o] * dy_scale * effective_scale[o];
        db[o] = dlinear_out;
        for (int i = 0; i < input_size; ++i) {
            int input_centered = (int)input[i] - input_zp;
            int weight_centered = (int)weight[o * input_size + i];
            dw[o * input_size + i] = dlinear_out * (float)input_centered;
            dx[i] += dlinear_out * (float)weight_centered;
        }
    }
}

static void ReluXQASBackward(const float *dy, const int8_t *relu_input,
                             int length, int act_min, float *dx) {
    for (int i = 0; i < length; ++i) {
        dx[i] = (relu_input[i] >= act_min && relu_input[i] <= 127) ? dy[i]
                                                                    : 0.0f;
    }
}

static void FlattenNCHWToNHWCBackward(const float *dy,
                                      float *dx, int height, int width,
                                      int channel) {
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channel; ++c) {
                dx[(h * width + w) * channel + c] =
                    dy[(c * height + h) * width + w];
            }
        }
    }
}

static void MaxPool2x2QASBackwardNHWC(const float *dy,
                                      const int8_t *pool_input, int input_h,
                                      int input_w, int channel,
                                      float *dx) {
    int output_h = input_h / 2;
    int output_w = input_w / 2;
    for (int oh = 0; oh < output_h; ++oh) {
        for (int ow = 0; ow < output_w; ++ow) {
            for (int c = 0; c < channel; ++c) {
                int base_h = oh * 2;
                int base_w = ow * 2;
                int max_index = (base_h * input_w + base_w) * channel + c;
                int8_t max_value = pool_input[max_index];
                for (int kh = 0; kh < 2; ++kh) {
                    for (int kw = 0; kw < 2; ++kw) {
                        int index =
                            ((base_h + kh) * input_w + base_w + kw) * channel +
                            c;
                        if (pool_input[index] > max_value) {
                            max_value = pool_input[index];
                            max_index = index;
                        }
                    }
                }
                dx[max_index] += dy[(oh * output_w + ow) * channel + c];
            }
        }
    }
}

static void Conv2d3x3ValidQASBackwardNHWC(
    const float *dy, const int8_t *input, const int8_t *weight,
    const float *effective_scale, int input_zp, int input_h, int input_w,
    int input_c, int output_c, float dy_scale, float *dx,
    float *dw, float *db) {
    int output_h = input_h - 2;
    int output_w = input_w - 2;
    for (int oh = 0; oh < output_h; ++oh) {
        for (int ow = 0; ow < output_w; ++ow) {
            for (int oc = 0; oc < output_c; ++oc) {
                float dlinear_out =
                    dy[(oh * output_w + ow) * output_c + oc] *
                    dy_scale * effective_scale[oc];
                db[oc] += dlinear_out;
                for (int kh = 0; kh < 3; ++kh) {
                    int ih = oh + kh;
                    for (int kw = 0; kw < 3; ++kw) {
                        int iw = ow + kw;
                        for (int ic = 0; ic < input_c; ++ic) {
                            int input_index =
                                (ih * input_w + iw) * input_c + ic;
                            int weight_index =
                                ((oc * 3 + kh) * 3 + kw) * input_c + ic;
                            int input_centered =
                                (int)input[input_index] - input_zp;
                            int weight_centered = (int)weight[weight_index];
                            dw[weight_index] +=
                                dlinear_out * (float)input_centered;
                            dx[input_index] +=
                                dlinear_out * (float)weight_centered;
                        }
                    }
                }
            }
        }
    }
}

static int RoundFloatToInt(float value) {
    return (int)(value >= 0.0f ? value + 0.5f : value - 0.5f);
}

static int8_t RoundClampInt8(float value) {
    int rounded = RoundFloatToInt(value);
    if (rounded > 127) {
        rounded = 127;
    } else if (rounded < -128) {
        rounded = -128;
    }
    return (int8_t)rounded;
}

static int32_t RoundClampInt32(float value) {
    if (value > 2147483647.0f) {
        return 2147483647;
    }
    if (value < -2147483648.0f) {
        return (int32_t)(-2147483647 - 1);
    }
    return (int32_t)RoundFloatToInt(value);
}

static void ZeroFloatBuffer(float *data, int length) {
    for (int i = 0; i < length; ++i) {
        data[i] = 0.0f;
    }
}

static void QASSGDUpdateInt8Weight(int8_t *weight, float *momentum,
                                   const float *d, const float *w_scale,
                                   int output_size, int inner_size,
                                   float learning_rate, float momentum_value) {
    for (int o = 0; o < output_size; ++o) {
        float inv_scale_sq = 1.0f / (w_scale[o] * w_scale[o]);
        for (int i = 0; i < inner_size; ++i) {
            int index = o * inner_size + i;
            float scaled_d = d[index] * inv_scale_sq;
            momentum[index] = momentum[index] * momentum_value + scaled_d;
            weight[index] = RoundClampInt8((float)weight[index] -
                                           learning_rate * momentum[index]);
        }
    }
}

static void QASSGDUpdateConv3x3RawWeight(int8_t *weight, float *momentum,
                                         const float *d,
                                         const float *w_scale, int output_c,
                                         int input_c, float learning_rate,
                                         float momentum_value) {
    for (int oc = 0; oc < output_c; ++oc) {
        float inv_scale_sq = 1.0f / (w_scale[oc] * w_scale[oc]);
        for (int ic = 0; ic < input_c; ++ic) {
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int weight_index = (oc * input_c + ic) * 9 + kh * 3 + kw;
                    int d_index = ((oc * 3 + kh) * 3 + kw) * input_c + ic;
                    float scaled_d = d[d_index] * inv_scale_sq;
                    momentum[weight_index] =
                        momentum[weight_index] * momentum_value + scaled_d;
                    weight[weight_index] =
                        RoundClampInt8((float)weight[weight_index] -
                                       learning_rate * momentum[weight_index]);
                }
            }
        }
    }
}

static void QASSGDUpdateInt32Bias(int32_t *bias, float *momentum,
                                  const float *d, const float *w_scale,
                                  float x_scale, int output_size,
                                  float learning_rate, float momentum_value) {
    for (int o = 0; o < output_size; ++o) {
        float bias_scale = x_scale * w_scale[o];
        float scaled_d = d[o] / (bias_scale * bias_scale);
        momentum[o] = momentum[o] * momentum_value + scaled_d;
        bias[o] = RoundClampInt32((float)bias[o] - learning_rate * momentum[o]);
    }
}

const unsigned char *m0_input0 = 0;
int m0_label0 = 0;

int SetInputs0(const void **inputs, int num) {
    if (inputs == NULL) {
        return RET_ERROR;
    }
    if (num != 2) {
        return RET_ERROR;
    }
    m0_input0 = (unsigned char *)inputs[0];
    m0_label0 = *((const int32_t *)inputs[1]);
    return RET_OK;
}
int GetBufferSize0() { return 16900 + 421276; }
int SetBuffer0(void *buffer) {
    m0_buffer = (unsigned char *)buffer;
    ZeroFloatBuffer((float *)(m0_buffer + 273192), 41246);
    return RET_OK;
}
void FreeResource0() {
    m0_buffer = NULL;
    m0_input0 = NULL;
    void **allocated[] = {

    };
    for (int i = 0; i < 0; ++i) {
        *(allocated[i]) = NULL;
    }
    if (m0_weight != NULL) {
        free(m0_weight);
        m0_weight = NULL;
    }
}
/*
m0_buffer timeline for current training graph.
Current graph: conv1 valid 1->12, conv2 valid 12->12, flatten 12*5*5=300.
Forward region size: 16900 bytes. Total training buffer: 438176 bytes.

Forward retained activations before CE/backward:
Pack input
0                                                 3136                                                                                                                                                                                                                                                                                                              16900
|------------------ fp32 input -------------------|----------------------------------------------------------------------------------------------------------------------------------------------------- free ------------------------------------------------------------------------------------------------------------------------------------------------------|

Quant input
0                                                 3136            3920                                                                                                                                                                                                                                                                                              16900
|------------------- old fp32 --------------------|--- q input ---|--------------------------------------------------------------------------------------------------------------------------------------------- free ----------------------------------------------------------------------------------------------------------------------------------------------|

Conv1 valid 1x28x28x1 -> 1x26x26x12
0                                                 3136            3920        3936                                      12048                                                                                                                                                                                                                                       16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|------------------------------------------------------------------------------------------------------------------ free -------------------------------------------------------------------------------------------------------------------|

Pool1 1x26x26x12 -> 1x13x13x12
0                                                 3136            3920        3936                                      12048       12064           14092                                                                                                                                                                                                           16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|--- gap ---|-- pool1 out --|---------------------------------------------------------------------------------------------------- free -----------------------------------------------------------------------------------------------------|

Conv2 valid 1x13x13x12 -> 1x11x11x12
0                                                 3136            3920        3936                                      12048       12064           14092       14112           15564                                                                                                                                                                               16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|--- gap ---|-- pool1 out --|--- gap ---|-- conv2 out --|-------------------------------------------------------------------------------------- free ---------------------------------------------------------------------------------------|

Pool2 1x11x11x12 -> 1x5x5x12
0                                                 3136            3920        3936                                      12048       12064           14092       14112           15564       15584         15884                                                                                                                                                     16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|--- gap ---|-- pool1 out --|--- gap ---|-- conv2 out --|--- gap ---|- pool2 out -|------------------------------------------------------------------------- free --------------------------------------------------------------------------|

Pack NHWC2NCHW
0                                                 3136            3920        3936                                      12048       12064           14092       14112           15564       15584         15884       15904         16204                                                                                                                           16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|--- gap ---|-- pool1 out --|--- gap ---|-- conv2 out --|--- gap ---|- pool2 out -|--- gap ---|- NCHW tmp --|------------------------------------------------------------ free -------------------------------------------------------------|

memcpy tmp -> FC1 input
0                                                 3136            3920        3936                                      12048       12064           14092       14112           15564       15584         15884       15904         16204       16224         16524                                                                                                 16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|--- gap ---|-- pool1 out --|--- gap ---|-- conv2 out --|--- gap ---|- pool2 out -|--- gap ---|-- old tmp --|--- gap ---|- FC1 input -|----------------------------------------------- free ------------------------------------------------|

FC1 input sums
0                                                 3136            3920        3936                                      12048       12064           14092       14112           15564       15584         15884       15904         16204       16224         16524                                                                                     16896       16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|--- gap ---|-- pool1 out --|--- gap ---|-- conv2 out --|--- gap ---|- pool2 out -|--- gap ---|-- old tmp --|--- gap ---|- FC1 input -|----------------------------------------- free ------------------------------------------|--- sum ---|

FC1 matmul 300 -> 128
0                                                 3136            3920        3936                                      12048       12064           14092       14112           15564       15584         15884       15904         16204       16224         16524       16544         16672                                                           16896       16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|--- gap ---|-- pool1 out --|--- gap ---|-- conv2 out --|--- gap ---|- pool2 out -|--- gap ---|-- old tmp --|--- gap ---|- FC1 input -|--- gap ---|-- FC1 out --|---------------------------- free -----------------------------|--- sum ---|

ReLU
0                                                 3136            3920        3936                                      12048       12064           14092       14112           15564       15584         15884       15904         16204       16224         16524       16544         16672         16800                                                         16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|--- gap ---|-- pool1 out --|--- gap ---|-- conv2 out --|--- gap ---|- pool2 out -|--- gap ---|-- old tmp --|--- gap ---|- FC1 input -|--- gap ---|-- old FC1 --|- ReLU out --|--------------------------- free ----------------------------|

FC2 input sums
0                                                 3136            3920        3936                                      12048       12064           14092       14112           15564       15584         15884       15904         16204       16224         16524       16544         16672         16800                                             16896       16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|--- gap ---|-- pool1 out --|--- gap ---|-- conv2 out --|--- gap ---|- pool2 out -|--- gap ---|-- old tmp --|--- gap ---|- FC1 input -|--- gap ---|-- old FC1 --|- FC2 input -|--------------------- free ----------------------|--- sum ---|

FC2 matmul 128 -> 10
0                                                 3136            3920        3936                                      12048       12064           14092       14112           15564       15584         15884       15904         16204       16224         16524       16544         16672         16800       16810                                 16896       16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|--- gap ---|-- pool1 out --|--- gap ---|-- conv2 out --|--- gap ---|- pool2 out -|--- gap ---|-- old tmp --|--- gap ---|- FC1 input -|--- gap ---|-- old FC1 --|- FC2 input -|- logits --|--------------- free ----------------|--- sum ---|

Dequant logits
0                                                 3136            3920        3936                                      12048       12064           14092       14112           15564       15584         15884       15904         16204       16224         16524       16544         16672         16800       16810       16832         16872       16896       16900
|------------------- old fp32 --------------------|--- q input ---|--- gap ---|--------------- conv1 out ---------------|--- gap ---|-- pool1 out --|--- gap ---|-- conv2 out --|--- gap ---|- pool2 out -|--- gap ---|-- old tmp --|--- gap ---|- FC1 input -|--- gap ---|-- old FC1 --|- FC2 input -| old int8 -|--- gap ---| fp32 logits |-- free ---|--- sum ---|

Total training allocation and backward/update timeline:
Total training allocation:
0                   16900                                                     17024                                             22696                                                                       178520          179720          185528          191336                                          204680          237128          269576                                          273192                        438176
|----- forward -----|-------------------------- CE ---------------------------|------------------- fc2 d -------------------|-------------------------------- fc1 d --------------------------------| flatten dx -|- pool2 dx --|- relu2 dx --|----------------- conv2 d -----------------|- pool1 dx --| conv1 dy|----------------- conv1 d -----------------|--------- momentum ----------|

Softmax + CrossEntropy:
0                   16900         16940         16980         16984           17024                                                                                                                                                                                                                                                                                                         273192                        438176
|----- forward -----|-- labels ---|--- prob ----|---- sum ----|- dlogits -|--------------------------------------------------------------------------------------------------------------------------------------------------- free ----------------------------------------------------------------------------------------------------------------------------------------------------|--------- momentum ----------|

FC2 backward:
0                   16900                                                     17024           17536             22656           22696                                                                                                                                                                                                                                                       273192                        438176
|----- forward -----|------------------------ CE old -------------------------|- fc2 dx -| fc2 dw | fc2 db |-------------------------------------------------------------------------------------------------------------------------- free ---------------------------------------------------------------------------------------------------------------------------|--------- momentum ----------|

ReLU backward for FC1 output:
0                   16900                                                                                                       22696           23208                                                                                                                                                                                                                                       273192                        438176
|----- forward -----|----------------------------------------------- prev grads ------------------------------------------------| relu3 dx -|------------------------------------------------------------------------------------------------------------------ free -------------------------------------------------------------------------------------------------------------------|--------- momentum ----------|

FC1 backward:
0                   16900                                                                                                       22696           23208           24408           24920                       178520                                                                                                                                                                          273192                        438176
|----- forward -----|----------------------------------------------- prev grads ------------------------------------------------| relu3 dx -|- fc1 dx -| fc1 db |----- fc1 dw -----|------------------------------------------------------------------------------------ free -------------------------------------------------------------------------------------|--------- momentum ----------|

Flatten backward:
0                   16900                                                                                                                                                                                   178520          179720                                                                                                                                                          273192                        438176
|----- forward -----|------------------------------------------------------------------------------------- prev grads --------------------------------------------------------------------------------------| flatten dx -|---------------------------------------------------------------------------- free -----------------------------------------------------------------------------|--------- momentum ----------|

Pool2 backward:
0                   16900                                                                                                                                                                                                   179720          185528                                                                                                                                          273192                        438176
|----- forward -----|--------------------------------------------------------------------------------------------- prev grads ----------------------------------------------------------------------------------------------| pool2 dx|-------------------------------------------------------------------- free ---------------------------------------------------------------------|--------- momentum ----------|

ReLU2 backward:
0                   16900                                                                                                                                                                                                                   185528          191336                                                                                                                          273192                        438176
|----- forward -----|----------------------------------------------------------------------------------------------------- prev grads ------------------------------------------------------------------------------------------------------| relu2 dx|------------------------------------------------------------ free -------------------------------------------------------------|--------- momentum ----------|

Conv2 backward:
0                   16900                                                                                                                                                                                                                                   191336          199448          204632          204680                                                                          273192                        438176
|----- forward -----|------------------------------------------------------------------------------------------------------------- prev grads --------------------------------------------------------------------------------------------------------------| conv2 dx| conv2 dw| conv2 db|------------------------------------ free -------------------------------------|--------- momentum ----------|

Pool1 backward:
0                   16900                                                                                                                                                                                                                                                                                   204680          237128                                                          273192                        438176
|----- forward -----|------------------------------------------------------------------------------------------------------------------------------------- prev grads --------------------------------------------------------------------------------------------------------------------------------------| pool1 dx|---------------------------- free -----------------------------|--------- momentum ----------|

ReLU1 backward:
0                   16900                                                                                                                                                                                                                                                                                                   237128          269576                                          273192                        438176
|----- forward -----|--------------------------------------------------------------------------------------------------------------------------------------------- prev grads ----------------------------------------------------------------------------------------------------------------------------------------------| conv1 dy|-------------------- free ---------------------|--------- momentum ----------|

Conv1 backward:
0                   16900                                                                                                                                                                                                                                                                                                                   269576          272712          273144          273192                        438176
|----- forward -----|----------------------------------------------------------------------------------------------------------------------------------------------------- prev grads ------------------------------------------------------------------------------------------------------------------------------------------------------| conv1 dx| conv1 dw| conv1 db|--------- momentum ----------|

QAS SGD update:
0                   16900                                                                                                                                                                                                                                                                                                                                                                   273192                        438176
|----- forward -----|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ all grads ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------- momentum ----------|
*/
void Execute0(bool train_mode) {
    {
        PackNCHWToNHWCFp32((float *)(m0_input0), (float *)(m0_buffer + 0), 1,
                           784, 1, 0, 1);
    }
    {
        DoQuantizeFp32ToInt8((float *)(m0_buffer + 0),
                             (int8_t *)(m0_buffer + 3136),
                             0.003921568859368562698, -128, 784, -128, 127);
    }
    {
        const int32_t unified_scale_int32[12] = {2993, 2918, 1572, 2898,
                                                 2190, 820,  2773, 2061,
                                                 1341, 1919, 2672, 2436};
        const int32_t input_shape[4] = {1, 28, 28, 1};
        const int32_t output_shape[4] = {1, 26, 26, 12};
        QuantArg conv_param__quant_arg_in[1] = {
            {0.003921568859368562698, -128}};
        QuantArg conv_param__quant_arg_w[12] = {
            {0.00240337359718978405, 0},  {0.002464865101501345634, 0},
            {0.004576579201966524124, 0}, {0.002482480835169553757, 0},
            {0.003284469712525606155, 0}, {0.008775365538895130157, 0},
            {0.002594307297840714455, 0}, {0.003490156028419733047, 0},
            {0.005364578217267990112, 0}, {0.003749208291992545128, 0},
            {0.002692127134650945663, 0}, {0.002952908864244818687, 0}};
        QuantArg conv_param__quant_arg_out[1] = {{0.0282081998884677887, -128}};
        double conv_param__real_multiplier[12] = {
            0.0003341225390599360766, 0.0003426712063512494144,
            0.0006362465548399112453, 0.0003451201956465963384,
            0.000456614531302144684,  0.001219971452530684832,
            0.0003606665832062876716, 0.0004852095346625980585,
            0.0007457960403941466454, 0.0005212235672482241075,
            0.0003742657024989233267, 0.0004105201974860015575};
        int32_t conv_param__left_shift[12] = {0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0};
        int32_t conv_param__right_shift[12] = {-11, -11, -10, -11, -11, -9,
                                               -11, -11, -10, -10, -11, -11};
        int32_t conv_param__quant_multiplier[12] = {
            1469486467, 1507083904, 1399120970, 1517854672,
            2008211946, 1341372798, 1586228408, 2133974101,
            1640022837, 1146182746, 1646037967, 1805486922};
        int32_t conv_param__out_act_min[1] = {-128};
        int32_t conv_param__out_act_max[1] = {127};
        ConvQuantArg conv_param__conv_quant_arg = {(RoundingMode)(1),
                                                   2,
                                                   conv_param__quant_arg_in,
                                                   conv_param__quant_arg_w,
                                                   conv_param__quant_arg_out,
                                                   conv_param__real_multiplier,
                                                   conv_param__left_shift,
                                                   conv_param__right_shift,
                                                   conv_param__quant_multiplier,
                                                   conv_param__out_act_min,
                                                   conv_param__out_act_max,
                                                   1,
                                                   12,
                                                   1,
                                                   2};
        int thread_num = MSMIN(m0_thread_num, 26);
        ConvParameter conv_param_ = {{"", 35, m0_thread_num, 0},
                                     conv_param__conv_quant_arg,
                                     3,
                                     3,
                                     1,
                                     1,
                                     1,
                                     1,
                                     0,
                                     0,
                                     0,
                                     0,
                                     1,
                                     0,
                                     1,
                                     28,
                                     28,
                                     1,
                                     1,
                                     26,
                                     26,
                                     12,
                                     thread_num,
                                     0,
                                     0,
                                     (PadType)(0),
                                     (ActType)(1),
                                     0,
                                     0,
                                     0};
        Conv3x3Int8LowMemory((int8_t *)(m0_buffer + 3136),
                             (int8_t *)(m0_buffer + 3936), m0_weight10,
                             m0_weight11, -128, unified_scale_int32, -128,
                             input_shape, output_shape, &conv_param_);
    }
    {
        const PoolingParameter pooling_parameter = {{"", 92, m0_thread_num, 0},
                                                    (PoolMode)(1),
                                                    (RoundType)(2),
                                                    (PadType)(0),
                                                    (ActType)(0),
                                                    0,
                                                    false,
                                                    false,
                                                    2,
                                                    2,
                                                    2,
                                                    2,
                                                    0,
                                                    0,
                                                    0,
                                                    0};
        PoolingComputeParam compute = {26, 26, 1, 12, 13,   13,
                                       1,  12, 2, 2,  -128, 127};
        static QuantArg quant_in = {0.0282081998884677887, -128};
        static QuantArg quant_out = {0.0282081998884677887, -128};
        static QuantArg *quant[2] = {&quant_in, &quant_out};
        MaxPooling2X2Int8((int8_t *)(m0_buffer + 3936),
                          (int8_t *)(m0_buffer + 12064), &pooling_parameter,
                          &compute, quant);
    }
    {
        const int32_t unified_scale_int32[12] = {912,  1873, 942,  1195,
                                                 1593, 1025, 1168, 1241,
                                                 1180, 4476, 1330, 1046};
        const int32_t input_shape[4] = {1, 13, 13, 12};
        const int32_t output_shape[4] = {1, 11, 11, 12};
        QuantArg conv_param__quant_arg_in[1] = {{0.0282081998884677887, -128}};
        QuantArg conv_param__quant_arg_w[12] = {
            {0.003704571863636374474, 0}, {0.001803990802727639675, 0},
            {0.003587289247661828995, 0}, {0.002827678108587861061, 0},
            {0.002120979130268096924, 0}, {0.003296336391940712929, 0},
            {0.002891680225729942322, 0}, {0.002722666598856449127, 0},
            {0.002864172216504812241, 0}, {0.0007547073764726519585, 0},
            {0.002540705492720007896, 0}, {0.003228385699912905693, 0}};
        QuantArg conv_param__quant_arg_out[1] = {
            {0.09529870003461837769, -128}};
        double conv_param__real_multiplier[12] = {
            0.00109654490675408639,   0.0005339772157570592866,
            0.001061829488073435645,  0.0008369863516137607436,
            0.0006278050229926450572, 0.0009757081550228583096,
            0.0008559308216028739232, 0.0008059031588439715711,
            0.0008477885093211708859, 0.0002233916852734197181,
            0.0007520430831742016663, 0.000955594858648127249};
        int32_t conv_param__left_shift[12] = {0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0};
        int32_t conv_param__right_shift[12] = {-9,  -10, -9,  -10, -10, -10,
                                               -10, -10, -10, -12, -10, -10};
        int32_t conv_param__quant_multiplier[12] = {
            1205663875, 1174228315, 1167493869, 1840552452,
            1380557846, 2145604924, 1882211782, 1772199788,
            1864306648, 1964974044, 1653760229, 2101375317};
        int32_t conv_param__out_act_min[1] = {-128};
        int32_t conv_param__out_act_max[1] = {127};
        ConvQuantArg conv_param__conv_quant_arg = {(RoundingMode)(1),
                                                   2,
                                                   conv_param__quant_arg_in,
                                                   conv_param__quant_arg_w,
                                                   conv_param__quant_arg_out,
                                                   conv_param__real_multiplier,
                                                   conv_param__left_shift,
                                                   conv_param__right_shift,
                                                   conv_param__quant_multiplier,
                                                   conv_param__out_act_min,
                                                   conv_param__out_act_max,
                                                   1,
                                                   12,
                                                   1,
                                                   2};
        int thread_num = MSMIN(m0_thread_num, 11);
        ConvParameter conv_param_ = {{"", 35, m0_thread_num, 0},
                                     conv_param__conv_quant_arg,
                                     3,
                                     3,
                                     1,
                                     1,
                                     1,
                                     1,
                                     0,
                                     0,
                                     0,
                                     0,
                                     1,
                                     0,
                                     1,
                                     13,
                                     13,
                                     12,
                                     1,
                                     11,
                                     11,
                                     12,
                                     thread_num,
                                     0,
                                     0,
                                     (PadType)(0),
                                     (ActType)(1),
                                     0,
                                     0,
                                     0};
        Conv3x3Int8LowMemory((int8_t *)(m0_buffer + 12064),
                             (int8_t *)(m0_buffer + 14112), m0_weight12,
                             m0_weight13, -128, unified_scale_int32, -128,
                             input_shape, output_shape, &conv_param_);
    }
    {
        const PoolingParameter pooling_parameter = {{"", 92, m0_thread_num, 0},
                                                    (PoolMode)(1),
                                                    (RoundType)(2),
                                                    (PadType)(0),
                                                    (ActType)(0),
                                                    0,
                                                    false,
                                                    false,
                                                    2,
                                                    2,
                                                    2,
                                                    2,
                                                    0,
                                                    0,
                                                    0,
                                                    0};
        PoolingComputeParam compute = {11, 11, 1, 12, 5,    5,
                                       1,  12, 2, 2,  -128, 127};
        static QuantArg quant_in = {0.09529870003461837769, -128};
        static QuantArg quant_out = {0.09529870003461837769, -128};
        static QuantArg *quant[2] = {&quant_in, &quant_out};
        MaxPooling2X2Int8((int8_t *)(m0_buffer + 14112),
                          (int8_t *)(m0_buffer + 15584), &pooling_parameter,
                          &compute, quant);
    }
    {
        PackNHWCToNCHWInt8((int8_t *)(m0_buffer + 15584),
                           (int8_t *)(m0_buffer + 15904), 1, 25, 12);
    }
    {
        memcpy((int8_t *)(m0_buffer + 16224), (int8_t *)(m0_buffer + 15904),
               300);
    }
    {
        int32_t tmp_weight_zp = 1;

        {
            CalcInputSums((int8_t *)(m0_buffer + 16224) + 0, 1, 300,
                          tmp_weight_zp, (int32_t *)(m0_buffer + 16896),
                          RowMajor);
            float filter_scale[128] = {
                0.0007036333554424345493, 0.00089493580162525177,
                0.0008662321488372981548, 0.0004847621312364935875,
                0.001038205460645258427,  0.0008613922400400042534,
                0.0005085899028927087784, 0.0004526878474280238152,
                0.000886210764292627573,  0.000450116029242053628,
                0.0008554600062780082226, 0.0004551475576590746641,
                0.0008021597168408334255, 0.0004565779236145317554,
                0.0008429869194515049458, 0.0007421507034450769424,
                0.0007129032746888697147, 0.001020946423523128033,
                0.0006989993853494524956, 0.0007197546074166893959,
                0.0009041049052029848099, 0.0007645745063200592995,
                0.001128330128267407417,  0.001007391023449599743,
                0.0009193703299388289452, 0.000469088525278493762,
                0.0004565624694805592299, 0.0005124008748680353165,
                0.0004535996995400637388, 0.001014999696053564548,
                0.001208808622322976589,  0.001015193993225693703,
                0.0006518071750178933144, 0.0008357270271517336369,
                0.000665930798277258873,  0.0006411027861759066582,
                0.0009029615321196615696, 0.0008670486276969313622,
                0.0006200710777193307877, 0.0007756930426694452763,
                0.001007999759167432785,  0.0004538916982710361481,
                0.001224888255819678307,  0.0008126370375975966454,
                0.0008304427610710263252, 0.0004580440581776201725,
                0.001010495005175471306,  0.0008452431648038327694,
                0.0009323265403509140015, 0.0009558276506140828133,
                0.001084820018149912357,  0.0005705995718017220497,
                0.0009654249879531562328, 0.0007167388102971017361,
                0.0006537092267535626888, 0.0008332137367688119411,
                0.0009587100357748568058, 0.0007566073909401893616,
                0.0009926271159201860428, 0.0008160088327713310719,
                0.001111041870899498463,  0.0009254722390323877335,
                0.0009574498399160802364, 0.0005007600993849337101,
                0.0009815618395805358887, 0.0007865872466936707497,
                0.001030452316626906395,  0.0006034672260284423828,
                0.0004716698895208537579, 0.0009541090112179517746,
                0.0007143935072235763073, 0.0006684918771497905254,
                0.0007137808715924620628, 0.0006947276415303349495,
                0.0006607567775063216686, 0.0007436206797137856483,
                0.0008575484971515834332, 0.0009387196041643619537,
                0.0007306224433705210686, 0.0007859048782847821712,
                0.0004595193604473024607, 0.001061097835190594196,
                0.0009493860416114330292, 0.000684030062984675169,
                0.0007996670319698750973, 0.0007386804209090769291,
                0.0008834503823891282082, 0.000771416176576167345,
                0.0009549440001137554646, 0.0009832112118601799011,
                0.0007799319573678076267, 0.0008150732610374689102,
                0.0005023335106670856476, 0.0008801249205134809017,
                0.001154426718130707741,  0.001334344269707798958,
                0.001064787618815898895,  0.0005329297855496406555,
                0.0009149283287115395069, 0.00101217546034604311,
                0.0006104715284891426563, 0.0005584904574789106846,
                0.000496453489176928997,  0.0004577803192660212517,
                0.000647868029773235321,  0.0004825000360142439604,
                0.000771309947595000267,  0.000720024982001632452,
                0.0009343706187792122364, 0.0005053246277384459972,
                0.0004647697496693581343, 0.0009870270732790231705,
                0.0004466344544198364019, 0.001001914613880217075,
                0.0004520487564150243998, 0.0008156223339028656483,
                0.0004673589428421109915, 0.0004573006881400942802,
                0.0008599355933256447315, 0.0004556145868264138699,
                0.0009623775840736925602, 0.001013883389532566071,
                0.0004657905083149671555, 0.0006434465176425874233,
                0.0009423660230822861195, 0.0006532659172080457211,
                0.0006988383247517049313, 0.001072134939022362232};
            int32_t filter_zp[128] = {
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int32_t left_shift[128] = {
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int32_t right_shift[128] = {
                -11, -10, -10, -11, -10, -10, -11, -11, -10, -11, -10, -11, -10,
                -11, -10, -10, -11, -10, -11, -11, -10, -10, -10, -10, -10, -11,
                -11, -11, -11, -10, -10, -10, -11, -10, -11, -11, -10, -10, -11,
                -10, -10, -11, -10, -10, -10, -11, -10, -10, -10, -10, -10, -11,
                -10, -11, -11, -10, -10, -10, -10, -10, -10, -10, -10, -11, -10,
                -10, -10, -11, -11, -10, -11, -11, -11, -11, -11, -10, -10, -10,
                -10, -10, -11, -10, -10, -11, -10, -10, -10, -10, -10, -10, -10,
                -10, -11, -10, -10, -10, -10, -11, -10, -10, -11, -11, -11, -11,
                -11, -11, -10, -11, -10, -11, -11, -10, -11, -10, -11, -10, -11,
                -11, -10, -11, -10, -10, -11, -11, -10, -11, -11, -10};
            int32_t multiplier[128] = {
                2095554195, 1332644753, 1289902207, 1443714049, 1545986853,
                1282695114, 1514677631, 1348190732, 1319652272, 1340531377,
                1273861445, 1355516178, 1194492240, 1359776062, 1255287877,
                1105133130, 2123161755, 1520286571, 2081753485, 2143566542,
                1346298348, 1138524260, 1680191108, 1500101208, 1369030046,
                1397035069, 1359730017, 1526027507, 1350906353, 1511431301,
                1800031202, 1511720645, 1941206073, 1244477238, 1983268976,
                1909326237, 1344595826, 1291118017, 1846689831, 1155080837,
                1501007665, 1351775975, 1823975413, 1210094042, 1236608464,
                1364142474, 1504723425, 1258647671, 1388323050, 1423318471,
                1615400373, 1699354277, 1437609871, 2134584734, 1946870608,
                1240734647, 1427610643, 1126660501, 1478116345, 1215114973,
                1654447283, 1378116336, 1425734061, 1491359060, 1461639124,
                1171303391, 1534441770, 1797240431, 1404722847, 1420759289,
                2127600020, 1990896270, 2125775509, 2069031247, 1967859539,
                1107322021, 1276971459, 1397842957, 1087966488, 1170287222,
                1368536286, 1580075817, 1413726257, 2037171875, 1190780460,
                1099965538, 1315541777, 1148712214, 1422002613, 1464095188,
                1161393069, 1213721804, 1496044944, 1310589857, 1719051452,
                1986965749, 1585570270, 1587166490, 1362415510, 1507225761,
                1818100550, 1663291071, 1478533136, 1363357098, 1929474537,
                1436977068, 1148554070, 2144371701, 1391366896, 1504953080,
                1384172877, 1469777345, 1330162540, 1491946387, 1346287320,
                1214539356, 1391884076, 1361928572, 1280526119, 1356907072,
                1433071899, 1509769027, 1387212972, 1916306403, 1403272834,
                1945550430, 2081273709, 1596511086};
            const MatmulQuantParameter matmul_quant_parameter = {
                {0.09529870003461837769, -128},
                {0, 0},
                {0.1407324671745300293, 19},
                -128,
                127,
                filter_scale,
                filter_zp,
                left_shift,
                right_shift,
                multiplier};
            int32_t *cur_left = matmul_quant_parameter.left_shift_ + 0;
            int32_t *cur_right = matmul_quant_parameter.right_shift_ + 0;
            int32_t *cur_mul = matmul_quant_parameter.quant_multiplier_ + 0;
            int32_t *cur_zp = matmul_quant_parameter.filter_zp_ + 0;
            MatmulInt8LowMemory(
                (int8_t *)(m0_buffer + 16224) + 0, m0_weight6 + 0 + 0,
                (int8_t *)(m0_buffer + 16544) + 0 + 0, 1, 128, 300,
                (int32_t *)(m0_buffer + 16896), m0_weight14 + 0 + 0, -128, 127,
                19, cur_mul, cur_left, cur_right, 128, true, cur_zp, true,
                false);
        }
    }
    {
        const ReluXQuantArg quant_arg = {
            {0.140732, 19}, {0.0596554, -128}, 1266526848, 2, 0, -128, 127};
        ReluXInt8((int8_t *)(m0_buffer + 16544), 128,
                  (int8_t *)(m0_buffer + 16672), &quant_arg);
    }
    {
        int32_t tmp_weight_zp = 1;

        {
            CalcInputSums((int8_t *)(m0_buffer + 16672) + 0, 1, 128,
                          tmp_weight_zp, (int32_t *)(m0_buffer + 16896),
                          RowMajor);
            float filter_scale[10] = {
                0.001879723509773612022, 0.001932796440087258816,
                0.002224280033260583878, 0.001817723736166954041,
                0.00209081755019724369,  0.002319189952686429024,
                0.002183472970500588417, 0.002257281215861439705,
                0.002566827693954110146, 0.002344903070479631424};
            int32_t filter_zp[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int32_t left_shift[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int32_t right_shift[10] = {-10, -10, -9, -10, -9,
                                       -9,  -9,  -9, -9,  -9};
            int32_t multiplier[10] = {
                1950345916, 2005412871, 1153923781, 1886016729, 1084685385,
                1203161644, 1132753684, 1171044316, 1331632474, 1216501280};
            const MatmulQuantParameter matmul_quant_parameter = {
                {0.05965540185570716858, -128},
                {0, 0},
                {0.1264334321022033691, -6},
                -128,
                127,
                filter_scale,
                filter_zp,
                left_shift,
                right_shift,
                multiplier};
            int32_t *cur_left = matmul_quant_parameter.left_shift_ + 0;
            int32_t *cur_right = matmul_quant_parameter.right_shift_ + 0;
            int32_t *cur_mul = matmul_quant_parameter.quant_multiplier_ + 0;
            int32_t *cur_zp = matmul_quant_parameter.filter_zp_ + 0;
            MatmulInt8LowMemory(
                (int8_t *)(m0_buffer + 16672) + 0, m0_weight8 + 0 + 0,
                (int8_t *)(m0_buffer + 16800) + 0 + 0, 1, 10, 128,
                (int32_t *)(m0_buffer + 16896), m0_weight15 + 0 + 0, -128, 127,
                -6, cur_mul, cur_left, cur_right, 10, true, cur_zp, true,
                false);
        }
    }
    {
        DoDequantizeInt8ToFp32((int8_t *)(m0_buffer + 16800),
                               (float *)(m0_buffer + 16832),
                               0.1264334321022033691, -6, 10);
    }
    if (!train_mode) {
        return;
    }
    {
        float *labels = (float *)(m0_buffer + 16900);
        float *prob = (float *)(m0_buffer + 16940);
        float *sum_data = (float *)(m0_buffer + 16980);
        float *dlogits = (float *)(m0_buffer + 16984);
        float loss = 0.0f;
        int input_shape[] = {1, 10, 0, 0, 0};
        for (int i = 0; i < 10; ++i) {
            labels[i] = 0.0f;
        }
        labels[m0_label0] = 1.0f;
        Softmax((float *)(m0_buffer + 16832), prob, sum_data, 1, 2,
                input_shape);
        ForwardPostExecute(labels, prob, dlogits, &loss, 10, 1);
        printf("1. Softmax && Cross Entropy\n   loss: %f\n", loss);
        PrintAndDumpGrads("ce_dlogits", dlogits, 10);
    }
    {
        float *fc2_dx = (float *)(m0_buffer + 17024);
        float *fc2_dw = (float *)(m0_buffer + 17536);
        float *fc2_db = (float *)(m0_buffer + 22656);
        const float filter_scale[10] = {
            0.001879723509773612022f, 0.001932796440087258816f,
            0.002224280033260583878f, 0.001817723736166954041f,
            0.00209081755019724369f,  0.002319189952686429024f,
            0.002183472970500588417f, 0.002257281215861439705f,
            0.002566827693954110146f, 0.002344903070479631424f};
        float effective_scale[10];
        for (int i = 0; i < 10; ++i) {
            effective_scale[i] = 0.05965540185570716858f * filter_scale[i] /
                                 0.1264334321022033691f;
        }
        ZeroFloatBuffer(fc2_dx, 128);
        FullyConnectedQASBackward(
            (float *)(m0_buffer + 16984), (int8_t *)(m0_buffer + 16672),
            m0_weight8, effective_scale, -128, 128, 10, 0.1264334321022033691f,
            fc2_dx, fc2_dw, fc2_db);
        printf("2. FullyConnectedQASBackward fc2\n");
        PrintAndDumpGrads("fc2_dx", fc2_dx, 128);
        PrintAndDumpGrads("fc2_dw", fc2_dw, 1280);
        PrintAndDumpGrads("fc2_db", fc2_db, 10);
    }
    {
        float *relu3_dx = (float *)(m0_buffer + 22696);
        ReluXQASBackward((float *)(m0_buffer + 17024),
                         (int8_t *)(m0_buffer + 16544), 128, 19,
                         relu3_dx);
        printf("3. ReluXQASBackward relu\n");
        PrintAndDumpGrads("relu3_dx", relu3_dx, 128);
    }
    {
        float *fc1_dx = (float *)(m0_buffer + 23208);
        float *fc1_db = (float *)(m0_buffer + 24408);
        float *fc1_dw = (float *)(m0_buffer + 24920);
        const float filter_scale[128] = {
            0.0007036333554424345493f, 0.00089493580162525177f,
            0.0008662321488372981548f, 0.0004847621312364935875f,
            0.001038205460645258427f,  0.0008613922400400042534f,
            0.0005085899028927087784f, 0.0004526878474280238152f,
            0.000886210764292627573f,  0.000450116029242053628f,
            0.0008554600062780082226f, 0.0004551475576590746641f,
            0.0008021597168408334255f, 0.0004565779236145317554f,
            0.0008429869194515049458f, 0.0007421507034450769424f,
            0.0007129032746888697147f, 0.001020946423523128033f,
            0.0006989993853494524956f, 0.0007197546074166893959f,
            0.0009041049052029848099f, 0.0007645745063200592995f,
            0.001128330128267407417f,  0.001007391023449599743f,
            0.0009193703299388289452f, 0.000469088525278493762f,
            0.0004565624694805592299f, 0.0005124008748680353165f,
            0.0004535996995400637388f, 0.001014999696053564548f,
            0.001208808622322976589f,  0.001015193993225693703f,
            0.0006518071750178933144f, 0.0008357270271517336369f,
            0.000665930798277258873f,  0.0006411027861759066582f,
            0.0009029615321196615696f, 0.0008670486276969313622f,
            0.0006200710777193307877f, 0.0007756930426694452763f,
            0.001007999759167432785f,  0.0004538916982710361481f,
            0.001224888255819678307f,  0.0008126370375975966454f,
            0.0008304427610710263252f, 0.0004580440581776201725f,
            0.001010495005175471306f,  0.0008452431648038327694f,
            0.0009323265403509140015f, 0.0009558276506140828133f,
            0.001084820018149912357f,  0.0005705995718017220497f,
            0.0009654249879531562328f, 0.0007167388102971017361f,
            0.0006537092267535626888f, 0.0008332137367688119411f,
            0.0009587100357748568058f, 0.0007566073909401893616f,
            0.0009926271159201860428f, 0.0008160088327713310719f,
            0.001111041870899498463f,  0.0009254722390323877335f,
            0.0009574498399160802364f, 0.0005007600993849337101f,
            0.0009815618395805358887f, 0.0007865872466936707497f,
            0.001030452316626906395f,  0.0006034672260284423828f,
            0.0004716698895208537579f, 0.0009541090112179517746f,
            0.0007143935072235763073f, 0.0006684918771497905254f,
            0.0007137808715924620628f, 0.0006947276415303349495f,
            0.0006607567775063216686f, 0.0007436206797137856483f,
            0.0008575484971515834332f, 0.0009387196041643619537f,
            0.0007306224433705210686f, 0.0007859048782847821712f,
            0.0004595193604473024607f, 0.001061097835190594196f,
            0.0009493860416114330292f, 0.000684030062984675169f,
            0.0007996670319698750973f, 0.0007386804209090769291f,
            0.0008834503823891282082f, 0.000771416176576167345f,
            0.0009549440001137554646f, 0.0009832112118601799011f,
            0.0007799319573678076267f, 0.0008150732610374689102f,
            0.0005023335106670856476f, 0.0008801244125134809017f,
            0.001154426718130707741f,  0.001334344269707798958f,
            0.001064787618815898895f,  0.0005329297855496406555f,
            0.0009149283287115395069f, 0.00101217546034604311f,
            0.0006104715284891426563f, 0.0005584904574789106846f,
            0.000496453489176928997f,  0.0004577803192660212517f,
            0.000647868029773235321f,  0.0004825000360142439604f,
            0.000771309947595000267f,  0.000720024982001632452f,
            0.0009343706187792122364f, 0.0005053246277384459972f,
            0.0004647697496693581343f, 0.0009870270732790231705f,
            0.0004466344544198364019f, 0.001001914613880217075f,
            0.0004520487564150243998f, 0.0008156223339028656483f,
            0.0004673589428421109915f, 0.0004573006881400942802f,
            0.0008599355933256447315f, 0.0004556145868264138699f,
            0.0009623775840736925602f, 0.001013883389532566071f,
            0.0004657905083149671555f, 0.0006434465176425874233f,
            0.0009423660230822861195f, 0.0006532659172080457211f,
            0.0006988383247517049313f, 0.001072134939022362232f};
        float effective_scale[128];
        for (int i = 0; i < 128; ++i) {
            effective_scale[i] = 0.09529870003461837769f * filter_scale[i] /
                                 0.1407324671745300293f;
        }
        ZeroFloatBuffer(fc1_dx, 300);
        FullyConnectedQASBackward(
            (float *)(m0_buffer + 22696), (int8_t *)(m0_buffer + 16224),
            m0_weight6, effective_scale, -128, 300, 128, 1.0f, fc1_dx,
            fc1_dw, fc1_db);
        printf("4. FullyConnectedQASBackward fc1\n");
        PrintAndDumpGrads("fc1_dx", fc1_dx, 300);
        PrintAndDumpGrads("fc1_dw", fc1_dw, 128 * 300);
        PrintAndDumpGrads("fc1_db", fc1_db, 128);
    }
    {
        float *flatten_dx = (float *)(m0_buffer + 178520);
        FlattenNCHWToNHWCBackward((float *)(m0_buffer + 23208),
                                  flatten_dx, 5, 5, 12);
        printf("5. FlattenQASBackward\n");
        PrintAndDumpGrads("flatten_dx", flatten_dx, 300);
    }
    {
        float *pool2_dx = (float *)(m0_buffer + 179720);
        ZeroFloatBuffer(pool2_dx, 11 * 11 * 12);
        MaxPool2x2QASBackwardNHWC((float *)(m0_buffer + 178520),
                                  (int8_t *)(m0_buffer + 14112), 11, 11, 12,
                                  pool2_dx);
        printf("6. MaxPool2x2QASBackward pool2\n");
        PrintAndDumpGrads("pool2_dx", pool2_dx, 11 * 11 * 12);
    }
    {
        float *relu2_dx = (float *)(m0_buffer + 185528);
        ReluXQASBackward((float *)(m0_buffer + 179720),
                         (int8_t *)(m0_buffer + 14112), 11 * 11 * 12, -128,
                         relu2_dx);
        printf("7. ReluXQASBackward relu2 (fused in conv2)\n");
        PrintAndDumpGrads("relu2_dx", relu2_dx, 11 * 11 * 12);
    }
    {
        float *conv2_dx = (float *)(m0_buffer + 191336);
        float *conv2_dw = (float *)(m0_buffer + 199448);
        float *conv2_db = (float *)(m0_buffer + 204632);
        const float filter_scale[12] = {
            0.003704571863636374474f, 0.001803990802727639675f,
            0.003587289247661828995f, 0.002827678108587861061f,
            0.002120979130268096924f, 0.003296336391940712929f,
            0.002891680225729942322f, 0.002722666598856449127f,
            0.002864172216504812241f, 0.0007547073764726519585f,
            0.002540705492720007896f, 0.003228385699912905693f};
        float effective_scale[12];
        for (int i = 0; i < 12; ++i) {
            effective_scale[i] = 0.0282081998884677887f * filter_scale[i] /
                                 0.09529870003461837769f;
        }
        ZeroFloatBuffer(conv2_dx, 13 * 13 * 12);
        ZeroFloatBuffer(conv2_dw, 12 * 3 * 3 * 12);
        ZeroFloatBuffer(conv2_db, 12);
        Conv2d3x3ValidQASBackwardNHWC(
            (float *)(m0_buffer + 185528), (int8_t *)(m0_buffer + 12064),
            m0_weight12, effective_scale, -128, 13, 13, 12, 12,
            0.09529870003461837769f, conv2_dx, conv2_dw,
            conv2_db);
        printf("8. Conv2d3x3ValidQASBackward conv2\n");
        PrintAndDumpGrads("conv2_dx", conv2_dx, 13 * 13 * 12);
        PrintAndDumpGrads("conv2_dw", conv2_dw,
                          12 * 3 * 3 * 12);
        PrintAndDumpGrads("conv2_db", conv2_db, 12);
    }
    {
        float *pool1_dx = (float *)(m0_buffer + 204680);
        ZeroFloatBuffer(pool1_dx, 26 * 26 * 12);
        MaxPool2x2QASBackwardNHWC((float *)(m0_buffer + 191336),
                                  (int8_t *)(m0_buffer + 3936), 26, 26, 12,
                                  pool1_dx);
        printf("9. MaxPool2x2QASBackward pool1\n");
        PrintAndDumpGrads("pool1_dx", pool1_dx, 26 * 26 * 12);
    }
    {
        float *conv1_dy = (float *)(m0_buffer + 237128);
        ReluXQASBackward((float *)(m0_buffer + 204680),
                         (int8_t *)(m0_buffer + 3936), 26 * 26 * 12, -128,
                         conv1_dy);
        printf("10. ReluXQASBackward relu1 (fused in conv1)\n");
        PrintAndDumpGrads("conv1_dy", conv1_dy, 26 * 26 * 12);
    }
    {
        float *conv1_dx = (float *)(m0_buffer + 269576);
        float *conv1_dw = (float *)(m0_buffer + 272712);
        float *conv1_db = (float *)(m0_buffer + 273144);
        const float filter_scale[12] = {
            0.00240337359718978405f,  0.002464865101501345634f,
            0.004576579201966524124f, 0.002482480835169553757f,
            0.003284469712525606155f, 0.008775365538895130157f,
            0.002594307297840714455f, 0.003490156028419733047f,
            0.005364578217267990112f, 0.003749208291992545128f,
            0.002692127134650945663f, 0.002952908864244818687f};
        float effective_scale[12];
        for (int i = 0; i < 12; ++i) {
            effective_scale[i] = 0.003921568859368562698f * filter_scale[i] /
                                 0.0282081998884677887f;
        }
        ZeroFloatBuffer(conv1_dx, 28 * 28);
        ZeroFloatBuffer(conv1_dw, 12 * 3 * 3);
        ZeroFloatBuffer(conv1_db, 12);
        Conv2d3x3ValidQASBackwardNHWC((float *)(m0_buffer + 237128),
                                      (int8_t *)(m0_buffer + 3136), m0_weight10,
                                      effective_scale, -128, 28, 28, 1, 12,
                                      0.0282081998884677887f, conv1_dx,
                                      conv1_dw, conv1_db);
        printf("11. Conv2d3x3ValidQASBackward conv1\n");
        PrintAndDumpGrads("conv1_dx", conv1_dx, 28 * 28);
        PrintAndDumpGrads("conv1_dw", conv1_dw, 12 * 3 * 3);
        PrintAndDumpGrads("conv1_db", conv1_db, 12);
    }
    {
        const float learning_rate = 0.00001f;
        const float momentum_value = 0.9f;
        float *momentum = (float *)(m0_buffer + 273192);

        {
            const float filter_scale[12] = {
                0.00240337359718978405f,  0.002464865101501345634f,
                0.004576579201966524124f, 0.002482480835169553757f,
                0.003284469712525606155f, 0.008775365538895130157f,
                0.002594307297840714455f, 0.003490156028419733047f,
                0.005364578217267990112f, 0.003749208291992545128f,
                0.002692127134650945663f, 0.002952908864244818687f};
            QASSGDUpdateConv3x3RawWeight(
                m0_weight10, momentum + 0, (float *)(m0_buffer + 272712),
                filter_scale, 12, 1, learning_rate, momentum_value);
            QASSGDUpdateInt32Bias(m0_weight11, momentum + 108,
                                  (float *)(m0_buffer + 273144), filter_scale,
                                  0.003921568859368562698f, 12, learning_rate,
                                  momentum_value);
            printf("12. QASSGDUpdate conv1\n");
        }
        {
            const float filter_scale[12] = {
                0.003704571863636374474f, 0.001803990802727639675f,
                0.003587289247661828995f, 0.002827678108587861061f,
                0.002120979130268096924f, 0.003296336391940712929f,
                0.002891680225729942322f, 0.002722666598856449127f,
                0.002864172216504812241f, 0.0007547073764726519585f,
                0.002540705492720007896f, 0.003228385699912905693f};
            QASSGDUpdateConv3x3RawWeight(
                m0_weight12, momentum + 120, (float *)(m0_buffer + 199448),
                filter_scale, 12, 12, learning_rate, momentum_value);
            QASSGDUpdateInt32Bias(m0_weight13, momentum + 1416,
                                  (float *)(m0_buffer + 204632), filter_scale,
                                  0.0282081998884677887f, 12, learning_rate,
                                  momentum_value);
            printf("13. QASSGDUpdate conv2\n");
        }
        {
            const float filter_scale[128] = {
                0.0007036333554424345493f, 0.00089493580162525177f,
                0.0008662321488372981548f, 0.0004847621312364935875f,
                0.001038205460645258427f,  0.0008613922400400042534f,
                0.0005085899028927087784f, 0.0004526878474280238152f,
                0.000886210764292627573f,  0.000450116029242053628f,
                0.0008554600062780082226f, 0.0004551475576590746641f,
                0.0008021597168408334255f, 0.0004565779236145317554f,
                0.0008429869194515049458f, 0.0007421507034450769424f,
                0.0007129032746888697147f, 0.001020946423523128033f,
                0.0006989993853494524956f, 0.0007197546074166893959f,
                0.0009041049052029848099f, 0.0007645745063200592995f,
                0.001128330128267407417f,  0.001007391023449599743f,
                0.0009193703299388289452f, 0.000469088525278493762f,
                0.0004565624694805592299f, 0.0005124008748680353165f,
                0.0004535996995400637388f, 0.001014999696053564548f,
                0.001208808622322976589f,  0.001015193993225693703f,
                0.0006518071750178933144f, 0.0008357270271517336369f,
                0.000665930798277258873f,  0.0006411027861759066582f,
                0.0009029615321196615696f, 0.0008670486276969313622f,
                0.0006200710777193307877f, 0.0007756930426694452763f,
                0.001007999759167432785f,  0.0004538916982710361481f,
                0.001224888255819678307f,  0.0008126370375975966454f,
                0.0008304427610710263252f, 0.0004580440581776201725f,
                0.001010495005175471306f,  0.0008452431648038327694f,
                0.0009323265403509140015f, 0.0009558276506140828133f,
                0.001084820018149912357f,  0.0005705995718017220497f,
                0.0009654249879531562328f, 0.0007167388102971017361f,
                0.0006537092267535626888f, 0.0008332137367688119411f,
                0.0009587100357748568058f, 0.0007566073909401893616f,
                0.0009926271159201860428f, 0.0008160088327713310719f,
                0.001111041870899498463f,  0.0009254722390323877335f,
                0.0009574498399160802364f, 0.0005007600993849337101f,
                0.0009815618395805358887f, 0.0007865872466936707497f,
                0.001030452316626906395f,  0.0006034672260284423828f,
                0.0004716698895208537579f, 0.0009541090112179517746f,
                0.0007143935072235763073f, 0.0006684918771497905254f,
                0.0007137808715924620628f, 0.0006947276415303349495f,
                0.0006607567775063216686f, 0.0007436206797137856483f,
                0.0008575484971515834332f, 0.0009387196041643619537f,
                0.0007306224433705210686f, 0.0007859048782847821712f,
                0.0004595193604473024607f, 0.001061097835190594196f,
                0.0009493860416114330292f, 0.000684030062984675169f,
                0.0007996670319698750973f, 0.0007386804209090769291f,
                0.0008834503823891282082f, 0.000771416176576167345f,
                0.0009549440001137554646f, 0.0009832112118601799011f,
                0.0007799319573678076267f, 0.0008150732610374689102f,
                0.0005023335106670856476f, 0.0008801244125134809017f,
                0.001154426718130707741f,  0.001334344269707798958f,
                0.001064787618815898895f,  0.0005329297855496406555f,
                0.0009149283287115395069f, 0.00101217546034604311f,
                0.0006104715284891426563f, 0.0005584904574789106846f,
                0.000496453489176928997f,  0.0004577803192660212517f,
                0.000647868029773235321f,  0.0004825000360142439604f,
                0.000771309947595000267f,  0.000720024982001632452f,
                0.0009343706187792122364f, 0.0005053246277384459972f,
                0.0004647697496693581343f, 0.0009870270732790231705f,
                0.0004466344544198364019f, 0.001001914613880217075f,
                0.0004520487564150243998f, 0.0008156223339028656483f,
                0.0004673589428421109915f, 0.0004573006881400942802f,
                0.0008599355933256447315f, 0.0004556145868264138699f,
                0.0009623775840736925602f, 0.001013883389532566071f,
                0.0004657905083149671555f, 0.0006434465176425874233f,
                0.0009423660230822861195f, 0.0006532659172080457211f,
                0.0006988383247517049313f, 0.001072134939022362232f};
            QASSGDUpdateInt8Weight(m0_weight6, momentum + 1428,
                                   (float *)(m0_buffer + 24920), filter_scale,
                                   128, 300, learning_rate, momentum_value);
            QASSGDUpdateInt32Bias(m0_weight14, momentum + 39828,
                                  (float *)(m0_buffer + 24408), filter_scale,
                                  0.09529870003461837769f, 128, learning_rate,
                                  momentum_value);
            printf("14. QASSGDUpdate fc1\n");
        }
        {
            const float filter_scale[10] = {
                0.001879723509773612022f, 0.001932796440087258816f,
                0.002224280033260583878f, 0.001817723736166954041f,
                0.00209081755019724369f,  0.002319189952686429024f,
                0.002183472970500588417f, 0.002257281215861439705f,
                0.002566827693954110146f, 0.002344903070479631424f};
            QASSGDUpdateInt8Weight(m0_weight8, momentum + 39956,
                                   (float *)(m0_buffer + 17536), filter_scale,
                                   10, 128, learning_rate, momentum_value);
            QASSGDUpdateInt32Bias(m0_weight15, momentum + 41236,
                                  (float *)(m0_buffer + 22656), filter_scale,
                                  0.05965540185570716858f, 10, learning_rate,
                                  momentum_value);
            printf("15. QASSGDUpdate fc2\n");
        }
    }
}
