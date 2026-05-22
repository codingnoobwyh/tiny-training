
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

#include "src/model0/net0.h"
#include "src/model0/weight0.h"

#define BENCHMARK

#ifdef BENCHMARK
#include <stdio.h>
static void PrintTop10Grads(const char *name, const float *data, int size)
{
    int print_size = size < 10 ? size : 10;
    printf("   %s:\t", name);
    for (int i = 0; i < print_size; ++i) {
        printf("%.3e\t", data[i]);
    }
    printf("\n");
}

static void DumpGradsToFile(const char *name, const float *data, int size)
{
    char file_name[128];
    snprintf(file_name, sizeof(file_name), "micro_grad_%s.txt", name);
    FILE *file = fopen(file_name, "w");
    if (file == NULL) {
        printf("   dump %s failed\n", file_name);
        return;
    }
    for (int i = 0; i < size; ++i) {
        fprintf(file, "%.9e\n", data[i]);
    }
    fclose(file);
}

static void PrintAndDumpGrads(const char *name, const float *data, int size)
{
    PrintTop10Grads(name, data, size);
    DumpGradsToFile(name, data, size);
}
#else
#define printf(...) ((void)0)
#define PrintTop10Grads(...) ((void)0)
#define DumpGradsToFile(...) ((void)0)
#define PrintAndDumpGrads(...) ((void)0)
#endif

const unsigned char *m0_input0 = 0;
static int m0_label0 = 0;

static const int kM0InferenceBufferSize = 66528;
static const int kM0CeBufferSize = 128;
static const int kM0QasBackwardBufferSize = 1007464;

static const int kM0Fc2GradInputOffset = 66656;
static const int kM0Fc2GradWeightOffset = 67168;
static const int kM0Fc2GradBiasOffset = 72288;
static const int kM0Fc1GradOutputOffset = 72328;
static const int kM0Fc1GradInputOffset = 72840;
static const int kM0Fc1GradBiasOffset = 79112;
static const int kM0Fc1GradWeightOffset = 79624;
static const int kM0FlattenGradOutputOffset = 882440;
static const int kM0Pool2GradInputOffset = 888712;
static const int kM0Relu2GradInputOffset = 913800;

/* Bind model input buffers. This training graph has image input plus sparse label. */
int SetInputs0(const void **inputs, int num) {
    if (inputs == NULL) {
        return RET_ERROR;
    }
    if (num != 2 || inputs[1] == NULL) {
        return RET_ERROR;
    }
    m0_input0 = (unsigned char *)inputs[0];
    m0_label0 = *((const int *)inputs[1]);
    if (m0_label0 < 0 || m0_label0 >= 10) {
        return RET_ERROR;
    }
    return RET_OK;
}

static void FullyConnectedQASBackward(const float *grad_output, const int8_t *input, const int8_t *weight,
                                      const float *effective_scale, int zero_x, int input_dim, int output_dim,
                                      float output_grad_scale, float *grad_input, float *grad_weight,
                                      float *grad_bias)
{
    for (int i = 0; i < input_dim; ++i) {
        grad_input[i] = 0.0f;
    }
    for (int o = 0; o < output_dim; ++o) {
        float grad_linear_out = grad_output[o] * output_grad_scale * effective_scale[o];
        if (grad_bias != NULL) {
            grad_bias[o] = grad_linear_out;
        }
        for (int i = 0; i < input_dim; ++i) {
            float input_centered = (float)input[i] - (float)zero_x;
            float grad_w = grad_linear_out * input_centered;
            if (grad_weight != NULL) {
                grad_weight[o * input_dim + i] = grad_w;
            }
            grad_input[i] += grad_linear_out * (float)weight[o * input_dim + i];
        }
    }
}

static void ReluXQASBackward(const float *grad_output, const int8_t *input, int input_dim, int zero_y,
                             float *grad_input)
{
    for (int i = 0; i < input_dim; ++i) {
        grad_input[i] = (input[i] >= zero_y && input[i] <= 127) ? grad_output[i] : 0.0f;
    }
}

static void FlattenNCHWToNHWCBackward(const float *grad_output_nchw, float *grad_input_nhwc, int height, int width,
                                      int channel)
{
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            for (int c = 0; c < channel; ++c) {
                grad_input_nhwc[(h * width + w) * channel + c] = grad_output_nchw[c * height * width + h * width + w];
            }
        }
    }
}

static void MaxPool2x2QASBackwardNHWC(const float *grad_output, const int8_t *input, int in_h, int in_w, int channel,
                                      float *grad_input)
{
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    for (int i = 0; i < in_h * in_w * channel; ++i) {
        grad_input[i] = 0.0f;
    }
    for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
            for (int c = 0; c < channel; ++c) {
                int base = ((h * 2) * in_w + (w * 2)) * channel + c;
                int idx0 = base;
                int idx1 = base + channel;
                int idx2 = base + in_w * channel;
                int idx3 = idx2 + channel;
                int max_idx = idx0;
                if (input[idx1] > input[max_idx]) {
                    max_idx = idx1;
                }
                if (input[idx2] > input[max_idx]) {
                    max_idx = idx2;
                }
                if (input[idx3] > input[max_idx]) {
                    max_idx = idx3;
                }
                grad_input[max_idx] += grad_output[(h * out_w + w) * channel + c];
            }
        }
    }
}

/* Runtime arena size used by m0_buffer.
 *
 * Buffer layout:
 *   ----------------------------- inference -----------------------------
 *   0     .. 3135    input_nhwc_fp32          3136   fp32, 1*28*28*4
 *   3136  .. 3919    input_int8               784    int8, 1*28*28
 *   3936  .. 16479   conv1_out_nhwc_int8      12544  int8, 1*28*28*16
 *   16480 .. 19615   pool1_out_int8           3136   int8, 1*14*14*16
 *   19616 .. 25887   conv2_out_nhwc_int8      6272   int8, 1*14*14*32
 *   25888 .. 27455   pool2_out_int8           1568   int8, 1*7*7*32
 *   27456 .. 29023   flatten_tmp_nchw_int8    1568   int8
 *   29024 .. 30591   fc1_input_int8_copy      1568   int8
 *   30592 .. 30719   fc1_out_int8             128    int8
 *   30720 .. 30847   relu3_out_int8           128    int8
 *   30848 .. 30857   fc2_out_int8             10     int8 logits code
 *   30880 .. 30919   logits_fp32              40     fp32 model output
 *   30944 .. 66527   operator workspace       35584  int8 conv/matmul temp buffers
 *   ----------------------------- inference -----------------------------
 *   ----------------------------- backward ------------------------------
 *   66528 .. 66567   ce_labels_fp32           40     fp32 one-hot label workspace
 *   66568 .. 66607   ce_prob_fp32             40     fp32 softmax workspace
 *   66608 .. 66611   ce_sum_data_fp32         4      fp32 softmax reduction workspace
 *   66612 .. 66651   ce_grad_logits_fp32      40     fp32, valid after Execute0(true)
 *   66652 .. 66655   ce_loss_fp32             4      fp32, valid after Execute0(true)
 *   66656 .. 67167   fc2_grad_input_fp32      512    fp32, 128
 *   67168 .. 72287   fc2_grad_weight_fp32     5120   fp32, 10*128
 *   72288 .. 72327   fc2_grad_bias_fp32       40     fp32, 10
 *   72328 .. 72839   fc1_grad_output_fp32     512    fp32, after ReLU backward
 *   72840 .. 79111   fc1_grad_input_fp32      6272   fp32, 1568
 *   79112 .. 79623   fc1_grad_bias_fp32       512    fp32, 128
 *   79624 .. 882439  fc1_grad_weight_fp32     802816 fp32, 128*1568
 *   882440.. 888711  flatten_grad_output_fp32 6272   fp32, NHWC 1*7*7*32
 *   888712.. 913799  pool2_grad_input_fp32    25088  fp32, NHWC 1*14*14*32
 *   913800.. 938887  relu2_grad_input_fp32    25088  fp32, NHWC 1*14*14*32
 *   ----------------------------- backward ------------------------------
 */
int GetBufferSize0() { 
    return kM0InferenceBufferSize + kM0CeBufferSize + kM0QasBackwardBufferSize;
}

int SetBuffer0(void *buffer) {
    m0_buffer = (unsigned char *)buffer;
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
        memset((int16_t *)(m0_buffer + 30944), 0, 2048);
        memset((int16_t *)(m0_buffer + 32992), 0, 256);
        memset((int32_t *)(m0_buffer + 33248), 0, 8192);
        memset((int8_t *)(m0_buffer + 41440), 0, 12544);
        memset((int16_t *)(m0_buffer + 53984), 0, 12544);
        QuantArg conv_param__quant_arg_in[1] = {
            {0.003921568859368562698, -128}};
        QuantArg conv_param__quant_arg_w[16] = {
            {0.002160259289667010307, 0}, {0.002494634361937642097, 0},
            {0.006594839971512556076, 0}, {0.002458167262375354767, 0},
            {0.00320643116720020771, 0},  {0.004997468087822198868, 0},
            {0.002034891629591584206, 0}, {0.002970720641314983368, 0},
            {0.00544007960706949234, 0},  {0.003800980513915419579, 0},
            {0.003385727526620030403, 0}, {0.002501456532627344131, 0},
            {0.002993785776197910309, 0}, {0.002527018077671527863, 0},
            {0.002368648303672671318, 0}, {0.007300233934074640274, 0}};
        QuantArg conv_param__quant_arg_out[1] = {
            {0.01951933093369007111, -128}};
        double conv_param__real_multiplier[16] = {
            0.0004340110710382293242, 0.0005011893418853257971,
            0.001324949025100385491,  0.0004938628593384653724,
            0.0006441942538886674197, 0.001004025987963388152,
            0.0004088238206836991117, 0.0005968383819663479082,
            0.001092949661959350359,  0.0007636433309841124525,
            0.0006802161380593721831, 0.0005025599675608799109,
            0.0006014723044808027979, 0.0005076954339552526563,
            0.000475877870424899947,  0.001466667600993276253};
        int32_t conv_param__left_shift[16] = {0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0};
        int32_t conv_param__right_shift[16] = {-11, -10, -9,  -10, -10, -9,
                                               -11, -10, -9,  -10, -10, -10,
                                               -10, -10, -11, -9};
        int32_t conv_param__quant_multiplier[16] = {
            1908800877, 1102127018, 1456796859, 1086015913,
            1416598145, 1103938248, 1798026178, 1312461482,
            1201710862, 1679269444, 1495811106, 1105141056,
            1322651585, 1116434066, 2092933008, 1612618081};
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
                                                   16,
                                                   1,
                                                   2};
        int thread_num = MSMIN(m0_thread_num, 28);
        ConvParameter conv_param_ = {{"", 35, m0_thread_num, 0},
                                     conv_param__conv_quant_arg,
                                     3,
                                     3,
                                     1,
                                     1,
                                     1,
                                     1,
                                     1,
                                     1,
                                     1,
                                     1,
                                     1,
                                     0,
                                     1,
                                     28,
                                     28,
                                     1,
                                     1,
                                     28,
                                     28,
                                     16,
                                     thread_num,
                                     0,
                                     0,
                                     (PadType)(0),
                                     (ActType)(1),
                                     0,
                                     0,
                                     0};
        PackInputToC8Int8((int8_t *)(m0_buffer + 3136),
                          (int16_t *)(m0_buffer + 53984), &conv_param_);
        Conv3x3Int8(
            (int16_t *)(m0_buffer + 53984), m0_weight10, m0_weight11,
            (int8_t *)(m0_buffer + 3936), (int16_t *)(m0_buffer + 30944),
            (int16_t *)(m0_buffer + 32992), (int32_t *)(m0_buffer + 33248),
            (int8_t *)(m0_buffer + 41440), 0, &conv_param_);
        PackNC4HW4ToNHWCInt8((int8_t *)(m0_buffer + 41440),
                             (int8_t *)(m0_buffer + 3936), 1, 784, 16);
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
        PoolingComputeParam compute = {28, 28, 1, 16, 14,   14,
                                       1,  16, 2, 2,  -128, 127};
        static QuantArg quant_in = {0.01951933093369007111, -128};
        static QuantArg quant_out = {0.01951933093369007111, -128};
        static QuantArg *quant[2] = {&quant_in, &quant_out};
        MaxPooling2X2Int8((int8_t *)(m0_buffer + 3936),
                          (int8_t *)(m0_buffer + 16480), &pooling_parameter,
                          &compute, quant);
    }
    {
        memset((int16_t *)(m0_buffer + 30944), 0, 4096);
        memset((int16_t *)(m0_buffer + 35040), 0, 256);
        memset((int32_t *)(m0_buffer + 35296), 0, 16384);
        memset((int8_t *)(m0_buffer + 51680), 0, 6272);
        memset((int16_t *)(m0_buffer + 57952), 0, 6272);
        QuantArg conv_param__quant_arg_in[1] = {{0.01951933093369007111, -128}};
        QuantArg conv_param__quant_arg_w[32] = {
            {0.001061098417267203331, 0},  {0.001755995675921440125, 0},
            {0.00147382635623216629, 0},   {0.001916126580908894539, 0},
            {0.001847039791755378246, 0},  {0.003189196810126304626, 0},
            {0.0006622434593737125397, 0}, {0.001426497474312782288, 0},
            {0.002257649321109056473, 0},  {0.001635198481380939484, 0},
            {0.001609747298061847687, 0},  {0.002540936926379799843, 0},
            {0.00109047209843993187, 0},   {0.001098028034903109074, 0},
            {0.00138328352477401495, 0},   {0.001591973588801920414, 0},
            {0.0006577332387678325176, 0}, {0.0006995517178438603878, 0},
            {0.001830157474614679813, 0},  {0.001479517435654997826, 0},
            {0.001479495316743850708, 0},  {0.0009992128470912575722, 0},
            {0.0008608305943198502064, 0}, {0.0006631871801801025867, 0},
            {0.0006408545887097716331, 0}, {0.001267497660592198372, 0},
            {0.002681609243154525757, 0},  {0.0008311981800943613052, 0},
            {0.001492832554504275322, 0},  {0.0008290793630294501781, 0},
            {0.001183047541417181492, 0},  {0.001331280916929244995, 0}};
        QuantArg conv_param__quant_arg_out[1] = {
            {0.07039272040128707886, -128}};
        double conv_param__real_multiplier[32] = {
            0.0002942340034644152125, 0.0004869233786940946372,
            0.0004086801145841701386, 0.0005313263492264090095,
            0.0005121691696696987909, 0.000884338469096368116,
            0.0001836347421677396018, 0.0003955561959032836993,
            0.0006260278725051071462, 0.0004534273076114598579,
            0.0004463698808047109686, 0.0007045811978504650918,
            0.0003023790865309049724, 0.0003044742673818243628,
            0.0003835733201403934245, 0.0004414413568183990369,
            0.0001823840963886640663, 0.0001939800222415619547,
            0.000507487836764156596,  0.0004102581993952313251,
            0.0004102520493348901787, 0.0002770736296337560165,
            0.0002387013359182985929, 0.0001838964298193134579,
            0.0001777037841864138054, 0.0003514668267671874121,
            0.0007435884982164386956, 0.0002304845193748888109,
            0.0004139503803688492707, 0.0002298969818875939324,
            0.0003280494900759434391, 0.0003691534183658578476};
        int32_t conv_param__left_shift[32] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                              0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int32_t conv_param__right_shift[32] = {
            -11, -11, -11, -10, -10, -10, -12, -11, -10, -11, -11,
            -10, -11, -11, -11, -11, -12, -12, -10, -11, -11, -11,
            -12, -12, -12, -11, -10, -12, -11, -12, -11, -11};
        int32_t conv_param__quant_multiplier[32] = {
            1294054832, 2141511667, 1797394152, 1168398998, 1126271915,
            1944680859, 1615268274, 1739674547, 1376649850, 1994194388,
            1963155497, 1549390439, 1329877287, 1339091989, 1686973302,
            1941479619, 1604267478, 1706266320, 1115977555, 1804334643,
            1804307594, 1218582710, 2099639155, 1617570103, 1563099016,
            1545767451, 1635168400, 2027363273, 1820573026, 2022195238,
            1442776915, 1623553904};
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
                                                   32,
                                                   1,
                                                   2};
        int thread_num = MSMIN(m0_thread_num, 14);
        ConvParameter conv_param_ = {{"", 35, m0_thread_num, 0},
                                     conv_param__conv_quant_arg,
                                     3,
                                     3,
                                     1,
                                     1,
                                     1,
                                     1,
                                     1,
                                     1,
                                     1,
                                     1,
                                     1,
                                     0,
                                     1,
                                     14,
                                     14,
                                     16,
                                     1,
                                     14,
                                     14,
                                     32,
                                     thread_num,
                                     0,
                                     0,
                                     (PadType)(0),
                                     (ActType)(1),
                                     0,
                                     0,
                                     0};
        PackInputToC8Int8((int8_t *)(m0_buffer + 16480),
                          (int16_t *)(m0_buffer + 57952), &conv_param_);
        Conv3x3Int8(
            (int16_t *)(m0_buffer + 57952), m0_weight12, m0_weight13,
            (int8_t *)(m0_buffer + 19616), (int16_t *)(m0_buffer + 30944),
            (int16_t *)(m0_buffer + 35040), (int32_t *)(m0_buffer + 35296),
            (int8_t *)(m0_buffer + 51680), 0, &conv_param_);
        PackNC4HW4ToNHWCInt8((int8_t *)(m0_buffer + 51680),
                             (int8_t *)(m0_buffer + 19616), 1, 196, 32);
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
        PoolingComputeParam compute = {14, 14, 1, 32, 7,    7,
                                       1,  32, 2, 2,  -128, 127};
        static QuantArg quant_in = {0.07039272040128707886, -128};
        static QuantArg quant_out = {0.07039272040128707886, -128};
        static QuantArg *quant[2] = {&quant_in, &quant_out};
        MaxPooling2X2Int8((int8_t *)(m0_buffer + 19616),
                          (int8_t *)(m0_buffer + 25888), &pooling_parameter,
                          &compute, quant);
    }
    {
        PackNHWCToNCHWInt8((int8_t *)(m0_buffer + 25888),
                           (int8_t *)(m0_buffer + 27456), 1, 49, 32);
    }
    {
        memcpy((int8_t *)(m0_buffer + 29024), (int8_t *)(m0_buffer + 27456),
               1568);
    }
    {
        int32_t tmp_weight_zp = 1;

        {
            CalcInputSums((int8_t *)(m0_buffer + 29024) + 0, 1, 1568,
                          tmp_weight_zp, (int32_t *)(m0_buffer + 30944),
                          RowMajor);
            float filter_scale[128] = {
                0.0004990499583072960377, 0.0004698942357208579779,
                0.0004256523388903588057, 0.0004050525021739304066,
                0.0005062564159743487835, 0.0007667241734452545643,
                0.0005676117725670337677, 0.0005347347469069063663,
                0.0002093825023621320724, 0.0005332875298336148262,
                0.0002091564965667203069, 0.0005324156372807919979,
                0.0008176540140993893147, 0.0007420015754178166389,
                0.000461498915683478117,  0.0007401778711937367916,
                0.0006282987887971103191, 0.0005517002427950501442,
                0.0004248088516760617495, 0.0004579625965561717749,
                0.0009474725229665637016, 0.0004808723751921206713,
                0.0006350801559165120125, 0.0005052032647654414177,
                0.000589806470088660717,  0.0005527743487618863583,
                0.0002258545137010514736, 0.0002115557726938277483,
                0.0006197291077114641666, 0.0005542998551391065121,
                0.0004904054803773760796, 0.0005884793936274945736,
                0.0006646378315053880215, 0.0006268147844821214676,
                0.000771204591728746891,  0.000343191495630890131,
                0.0004469733394216746092, 0.0005330626736395061016,
                0.0005322563811205327511, 0.0006090266397222876549,
                0.0003732676268555223942, 0.0006146946107037365437,
                0.0005256849690340459347, 0.0004067923582624644041,
                0.0005288260872475802898, 0.0004081292427144944668,
                0.000387039093766361475,  0.0004967917921021580696,
                0.0007125694537535309792, 0.0003341789415571838617,
                0.0006017450941726565361, 0.0007151186582632362843,
                0.0005828648572787642479, 0.0008780374773778021336,
                0.000598438724409788847,  0.0004735869006253778934,
                0.0004734734829980880022, 0.000454585155239328742,
                0.0002173963439418002963, 0.0004896494210697710514,
                0.0004519391222856938839, 0.0007707824697718024254,
                0.0006946329958736896515, 0.0005095974775031208992,
                0.0004898164770565927029, 0.0004857347230426967144,
                0.0006825198070146143436, 0.0003490003873594105244,
                0.000595043704379349947,  0.0007196873193606734276,
                0.0006500243325717747211, 0.0006580259068869054317,
                0.0006231302977539598942, 0.0005507636233232915401,
                0.0004763811593875288963, 0.0007247324101626873016,
                0.0006393145304173231125, 0.0007146944408304989338,
                0.0003687126445583999157, 0.0004604405839927494526,
                0.0004608354647643864155, 0.0008357622427865862846,
                0.0003225939872208982706, 0.0003279900411143898964,
                0.0001988275907933712006, 0.0005219030426815152168,
                0.0007622330449521541595, 0.0007301504374481737614,
                0.0007964133983477950096, 0.0005391595186665654182,
                0.0004630415351130068302, 0.0004879671614617109299,
                0.0004524565592873841524, 0.0002600255538709461689,
                0.0009644408710300922394, 0.0006423963350243866444,
                0.000461512798210605979,  0.0004835981817450374365,
                0.000609989918302744627,  0.0005962647264823317528,
                0.0006453283713199198246, 0.0002007585426326841116,
                0.000528725737240165472,  0.0005231396644376218319,
                0.0009288404835388064384, 0.0006351047195494174957,
                0.0008702647173777222633, 0.0006170542328618466854,
                0.0009385936427861452103, 0.0004415948933456093073,
                0.0002234724815934896469, 0.0006334488280117511749,
                0.0004916627076454460621, 0.0006044852780178189278,
                0.0004003567155450582504, 0.000593172968365252018,
                0.0004656077071558684111, 0.0008042655535973608494,
                0.0006218030466698110104, 0.00096041080541908741,
                0.0003428751660976558924, 0.0007565841660834848881,
                0.0002005088463192805648, 0.0004907636321149766445,
                0.0007347305072471499443, 0.0002462097618263214827,
                0.0006546863587573170662, 0.0002161292650271207094};
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
                -11, -12, -12, -12, -11, -11, -11, -11, -13, -11, -13, -11, -11,
                -11, -12, -11, -11, -11, -12, -12, -10, -11, -11, -11, -11, -11,
                -13, -13, -11, -11, -11, -11, -11, -11, -11, -12, -12, -11, -11,
                -11, -12, -11, -11, -12, -11, -12, -12, -11, -11, -12, -11, -11,
                -11, -11, -11, -11, -11, -12, -13, -11, -12, -11, -11, -11, -11,
                -11, -11, -12, -11, -11, -11, -11, -11, -11, -11, -11, -11, -11,
                -12, -12, -12, -11, -12, -12, -13, -11, -11, -11, -11, -11, -12,
                -11, -12, -12, -10, -11, -12, -11, -11, -11, -11, -13, -11, -11,
                -11, -11, -11, -11, -11, -12, -13, -11, -11, -11, -12, -11, -12,
                -11, -11, -10, -12, -11, -13, -11, -11, -12, -11, -13};
            int32_t multiplier[128] = {
                1137111436, 2141357156, 1939742277, 1845866678, 1153531767,
                1747021209, 1293333092, 1218421050, 1908355930, 1215123583,
                1906296102, 1213136883, 1863067511, 1690689421, 2103098907,
                1686533967, 1431611684, 1257077911, 1935898412, 2086983452,
                1079432846, 1095692908, 1447063336, 1151132089, 1343904853,
                1259525281, 2058485244, 1928163577, 1412085243, 1263001270,
                1117414599, 1340881056, 1514412118, 1428230256, 1757230027,
                1563959617, 2036904392, 1214611216, 1212774069, 1387699036,
                1701019647, 1400613821, 1197800752, 1853795343, 1204957864,
                1859887685, 1763777623, 1131966106, 1623626791, 1522888358,
                1371107720, 1629435336, 1328088033, 2000654392, 1363573899,
                1079092524, 1078834163, 2071592210, 1981395733, 1115691795,
                2059533999, 1756268178, 1582757606, 1161144486, 1116072508,
                1106771959, 1555157069, 1590431323, 1355838240, 1639845167,
                1481114436, 1499346478, 1419835033, 1254943778, 1085459361,
                1651340730, 1456711625, 1628468658, 1680262161, 2098275999,
                2100075345, 1904328008, 1470094617, 1494684899, 1812156253,
                1189183437, 1736787898, 1663686036, 1814669568, 1228503160,
                2110128840, 1111858763, 2061891990, 1184963687, 1098764515,
                1463733668, 2103162025, 1101903715, 1389893933, 1358620397,
                1470414448, 1829755345, 1204729295, 1192001157, 2116411715,
                1447119389, 1982943783, 1405990311, 2138634820, 2012394185,
                2036775093, 1443346296, 1120279186, 1377351381, 1824467410,
                1351575627, 2121822943, 1832561171, 1416810765, 1094173120,
                1562518020, 1723916566, 1827479549, 1118230664, 1674121892,
                1122003639, 1491737174, 1969847415};
            const MatmulQuantParameter matmul_quant_parameter = {
                {0.07039272040128707886, -128},
                {0, 0},
                {0.1358715593814849854, 12},
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
                (int8_t *)(m0_buffer + 29024) + 0, m0_weight6 + 0 + 0,
                (int8_t *)(m0_buffer + 30592) + 0 + 0, 1, 128, 1568,
                (int32_t *)(m0_buffer + 30944), m0_weight14 + 0 + 0, -128, 127,
                12, cur_mul, cur_left, cur_right, 128, true, cur_zp, true,
                false);
        }
    }
    {
        const ReluXQuantArg quant_arg = {
            {0.135872, 12}, {0.0612587, -128}, 1190777728, 2, 0, -128, 127};
        ReluXInt8((int8_t *)(m0_buffer + 30592), 128,
                  (int8_t *)(m0_buffer + 30720), &quant_arg);
    }
    {
        int32_t tmp_weight_zp = 1;

        {
            CalcInputSums((int8_t *)(m0_buffer + 30720) + 0, 1, 128,
                          tmp_weight_zp, (int32_t *)(m0_buffer + 30944),
                          RowMajor);
            float filter_scale[10] = {
                0.001925971242599189281, 0.001776357647031545639,
                0.002733679721131920815, 0.00220415974035859108,
                0.001976877916604280472, 0.002213228959590196609,
                0.002378750592470169067, 0.001968011027202010155,
                0.00197996385395526886,  0.001833997899666428566};
            int32_t filter_zp[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int32_t left_shift[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int32_t right_shift[10] = {-10, -10, -9,  -9,  -10,
                                       -9,  -9,  -10, -10, -10};
            int32_t multiplier[10] = {
                1971153685, 1818030141, 1398905333, 1127934182, 2023254591,
                1132575126, 1217277512, 2014179600, 2026412859, 1877022687};
            const MatmulQuantParameter matmul_quant_parameter = {
                {0.06125869229435920715, -128},
                {0, 0},
                {0.1316215097904205322, -5},
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
                (int8_t *)(m0_buffer + 30720) + 0, m0_weight8 + 0 + 0,
                (int8_t *)(m0_buffer + 30848) + 0 + 0, 1, 10, 128,
                (int32_t *)(m0_buffer + 30944), m0_weight15 + 0 + 0, -128, 127,
                -5, cur_mul, cur_left, cur_right, 10, true, cur_zp, true,
                false);
        }
    }
    {
        DoDequantizeInt8ToFp32((int8_t *)(m0_buffer + 30848),
                               (float *)(m0_buffer + 30880),
                               0.1316215097904205322, -5, 10);
    }
    /* ===== Training loss: fp32 logits -> softmax probabilities -> cross entropy ===== */
    printf("\n[*] Training:\n");
    {
        float *labels = (float *)(m0_buffer + 66528);
        float *prob = (float *)(m0_buffer + 66568);
        float *sum_data = (float *)(m0_buffer + 66608);
        float *grad_logits = (float *)(m0_buffer + 66612);
        float *loss = (float *)(m0_buffer + 66652);
        int input_shape[] = {1, 10, 0, 0, 0};
        for (int i = 0; i < 10; ++i) {
            labels[i] = 0.0f;
        }
        labels[m0_label0] = 1.0f;
        Softmax((float *)(m0_buffer + 30880), prob, sum_data, 1, 2, input_shape);
        ForwardPostExecute(labels, prob, grad_logits, loss, 10, 1);
        printf("1. Softmax && Cross Entropy\n   loss: %f\n", loss[0]);
        PrintAndDumpGrads("ce_grad_logits", grad_logits, 10);
    }
    {
        const float fc2_filter_scale[10] = {
            0.001925971242599189281f, 0.001776357647031545639f,
            0.002733679721131920815f, 0.00220415974035859108f,
            0.001976877916604280472f, 0.002213228959590196609f,
            0.002378750592470169067f, 0.001968011027202010155f,
            0.00197996385395526886f, 0.001833997899666428566f};
        float fc2_effective_scale[10];
        for (int i = 0; i < 10; ++i) {
            fc2_effective_scale[i] = 0.06125869229435920715f * fc2_filter_scale[i] / 0.1316215097904205322f;
        }
        FullyConnectedQASBackward((float *)(m0_buffer + 66612), (int8_t *)(m0_buffer + 30720), m0_weight8,
                                  fc2_effective_scale, -128, 128, 10, 0.1316215097904205322f,
                                  (float *)(m0_buffer + kM0Fc2GradInputOffset),
                                  (float *)(m0_buffer + kM0Fc2GradWeightOffset),
                                  (float *)(m0_buffer + kM0Fc2GradBiasOffset));
        printf("2. FullyConnectedQASBackward fc2\n");
        PrintAndDumpGrads("fc2_grad_input", (float *)(m0_buffer + kM0Fc2GradInputOffset), 128);
        PrintAndDumpGrads("fc2_grad_weight", (float *)(m0_buffer + kM0Fc2GradWeightOffset), 1280);
        PrintAndDumpGrads("fc2_grad_bias", (float *)(m0_buffer + kM0Fc2GradBiasOffset), 10);
    }
    {
        ReluXQASBackward((float *)(m0_buffer + kM0Fc2GradInputOffset), (int8_t *)(m0_buffer + 30592), 128, 12,
                         (float *)(m0_buffer + kM0Fc1GradOutputOffset));
        printf("3. ReluXQASBackward fc1\n");
        PrintAndDumpGrads("fc1_grad_output", (float *)(m0_buffer + kM0Fc1GradOutputOffset), 128);
    }
    {
        const float fc1_filter_scale[128] = {
            0.0004990499583072960377f, 0.0004698942357208579779f,
            0.0004256523388903588057f, 0.0004050525021739304066f,
            0.0005062564159743487835f, 0.0007667241734452545643f,
            0.0005676117725670337677f, 0.0005347347469069063663f,
            0.0002093825023621320724f, 0.0005332875298336148262f,
            0.0002091564965667203069f, 0.0005324156372807919979f,
            0.0008176540140993893147f, 0.0007420015754178166389f,
            0.000461498915683478117f, 0.0007401778711937367916f,
            0.0006282987887971103191f, 0.0005517002427950501442f,
            0.0004248088516760617495f, 0.0004579625965561717749f,
            0.0009474725229665637016f, 0.0004808723751921206713f,
            0.0006350801559165120125f, 0.0005052032647654414177f,
            0.000589806470088660717f,  0.0005527743487618863583f,
            0.0002258545137010514736f, 0.0002115557726938277483f,
            0.0006197291077114641666f, 0.0005542998551391065121f,
            0.0004904054803773760796f, 0.0005884793936274945736f,
            0.0006646378315053880215f, 0.0006268147844821214676f,
            0.000771204591728746891f,  0.000343191495630890131f,
            0.0004469733394216746092f, 0.0005330626736395061016f,
            0.0005322563811205327511f, 0.0006090266397222876549f,
            0.0003732676268555223942f, 0.0006146946107037365437f,
            0.0005256849690340459347f, 0.0004067923582624644041f,
            0.0005288260872475802898f, 0.0004081292427144944668f,
            0.000387039093766361475f,  0.0004967917921021580696f,
            0.0007125694537535309792f, 0.0003341789415571838617f,
            0.0006017450941726565361f, 0.0007151186582632362843f,
            0.0005828648572787642479f, 0.0008780374773778021336f,
            0.000598438724409788847f,  0.0004735869006253778934f,
            0.0004734734829980880022f, 0.000454585155239328742f,
            0.0002173963439418002963f, 0.0004896494210697710514f,
            0.0004519391222856938839f, 0.0007707824697718024254f,
            0.0006946329958736896515f, 0.0005095974775031208992f,
            0.0004898164770565927029f, 0.0004857347230426967144f,
            0.0006825198070146143436f, 0.0003490003873594105244f,
            0.000595043704379349947f,  0.0007196873193606734276f,
            0.0006500243325717747211f, 0.0006580259068869054317f,
            0.0006231302977539598942f, 0.0005507636233232915401f,
            0.0004763811593875288963f, 0.0007247324101626873016f,
            0.0006393145304173231125f, 0.0007146944408304989338f,
            0.0003687126445583999157f, 0.0004604405839927494526f,
            0.0004608354647643864155f, 0.0008357622427865862846f,
            0.0003225939872208982706f, 0.0003279900411143898964f,
            0.0001988275907933712006f, 0.0005219030426815152168f,
            0.0007622330449521541595f, 0.0007301504374481737614f,
            0.0007964133983477950096f, 0.0005391595186665654182f,
            0.0004630415351130068302f, 0.0004879671614617109299f,
            0.0004524565592873841524f, 0.0002600255538709461689f,
            0.0009644408710300922394f, 0.0006423963350243866444f,
            0.000461512798210605979f,  0.0004835981817450374365f,
            0.000609989918302744627f,  0.0005962647264823317528f,
            0.0006453283713199198246f, 0.0002007585426326841116f,
            0.000528725737240165472f,  0.0005231396644376218319f,
            0.0009288404835388064384f, 0.0006351047195494174957f,
            0.0008702647173777222633f, 0.0006170542328618466854f,
            0.0009385936427861452103f, 0.0004415948933456093073f,
            0.0002234724815934896469f, 0.0006334488280117511749f,
            0.0004916627076454460621f, 0.0006044852780178189278f,
            0.0004003567155450582504f, 0.000593172968365252018f,
            0.0004656077071558684111f, 0.0008042655535973608494f,
            0.0006218030466698110104f, 0.00096041080541908741f,
            0.0003428751660976558924f, 0.0007565841660834848881f,
            0.0002005088463192805648f, 0.0004907636321149766445f,
            0.0007347305072471499443f, 0.0002462097618263214827f,
            0.0006546863587573170662f, 0.0002161292650271207094f};
        float fc1_effective_scale[128];
        for (int i = 0; i < 128; ++i) {
            fc1_effective_scale[i] = 0.07039272040128707886f * fc1_filter_scale[i] / 0.1358715593814849854f;
        }
        FullyConnectedQASBackward((float *)(m0_buffer + kM0Fc1GradOutputOffset), (int8_t *)(m0_buffer + 29024),
                                  m0_weight6, fc1_effective_scale, -128, 1568, 128, 1.0f,
                                  (float *)(m0_buffer + kM0Fc1GradInputOffset),
                                  (float *)(m0_buffer + kM0Fc1GradWeightOffset),
                                  (float *)(m0_buffer + kM0Fc1GradBiasOffset));
        printf("4. FullyConnectedQASBackward fc1\n");
        PrintAndDumpGrads("fc1_grad_input", (float *)(m0_buffer + kM0Fc1GradInputOffset), 1568);
        PrintAndDumpGrads("fc1_grad_weight", (float *)(m0_buffer + kM0Fc1GradWeightOffset), 128 * 1568);
        PrintAndDumpGrads("fc1_grad_bias", (float *)(m0_buffer + kM0Fc1GradBiasOffset), 128);
    }
    {
        FlattenNCHWToNHWCBackward((float *)(m0_buffer + kM0Fc1GradInputOffset),
                                  (float *)(m0_buffer + kM0FlattenGradOutputOffset), 7, 7, 32);
        printf("5. FlattenQASBackward\n");
        PrintAndDumpGrads("flatten_grad_output", (float *)(m0_buffer + kM0FlattenGradOutputOffset), 1568);
    }
    {
        MaxPool2x2QASBackwardNHWC((float *)(m0_buffer + kM0FlattenGradOutputOffset), (int8_t *)(m0_buffer + 19616),
                                  14, 14, 32, (float *)(m0_buffer + kM0Pool2GradInputOffset));
        printf("6. MaxPool2x2QASBackward pool2\n");
        PrintAndDumpGrads("pool2_grad_input", (float *)(m0_buffer + kM0Pool2GradInputOffset), 14 * 14 * 32);
    }
    {
        ReluXQASBackward((float *)(m0_buffer + kM0Pool2GradInputOffset), (int8_t *)(m0_buffer + 19616),
                         14 * 14 * 32, -128, (float *)(m0_buffer + kM0Relu2GradInputOffset));
        printf("7. ReluXQASBackward relu2\n");
        PrintAndDumpGrads("relu2_grad_input", (float *)(m0_buffer + kM0Relu2GradInputOffset), 14 * 14 * 32);
    }
}
