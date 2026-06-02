TOTAL_SIZE = 12068

SEGMENT_GROUPS = [
    {
        "title": "Inference reuse forward timeline",
        "total_size": TOTAL_SIZE,
        "rows": [
            {
                "name": "Pack input",
                "segments": [
                    ("fp32 input", 0, 3136),
                    ("free", 3136, 12068),
                ],
            },
            {
                "name": "Quant input",
                "segments": [
                    ("fp32 input", 0, 3136),
                    ("q input", 3136, 3920),
                    ("free", 3920, 12068),
                ],
            },
            {
                "name": "Conv1",
                "segments": [
                    ("old fp32", 0, 3136),
                    ("q input", 3136, 3920),
                    ("gap", 3920, 3936),
                    ("conv1 out", 3936, 12048),
                    ("free", 12048, 12068),
                ],
            },
            {
                "name": "Pool1",
                "segments": [
                    ("pool1 out", 0, 2028),
                    ("dead fp32", 2028, 3136),
                    ("dead q", 3136, 3920),
                    ("gap", 3920, 3936),
                    ("old conv1", 3936, 12048),
                    ("free", 12048, 12068),
                ],
            },
            {
                "name": "Conv2",
                "segments": [
                    ("conv2 in", 0, 2028),
                    ("gap", 2028, 2048),
                    ("conv2 out", 2048, 3500),
                    ("free", 3500, 12068),
                ],
            },
            {
                "name": "Pool2",
                "segments": [
                    ("pool2 out", 0, 300),
                    ("dead", 300, 2048),
                    ("old conv2", 2048, 3500),
                    ("free", 3500, 12068),
                ],
            },
            {
                "name": "NHWC2NCHW",
                "segments": [
                    ("NHWC", 0, 300),
                    ("gap", 300, 320),
                    ("NCHW tmp", 320, 620),
                    ("free", 620, 12068),
                ],
            },
            {
                "name": "FC1 memcpy",
                "segments": [
                    ("FC1 input", 0, 300),
                    ("gap", 300, 320),
                    ("old tmp", 320, 620),
                    ("free", 620, 12068),
                ],
            },
            {
                "name": "FC1 sums",
                "segments": [
                    ("FC1 input", 0, 300),
                    ("free", 300, 12064),
                    ("sum", 12064, 12068),
                ],
            },
            {
                "name": "FC1 matmul",
                "segments": [
                    ("FC1 input", 0, 300),
                    ("gap", 300, 320),
                    ("FC1 out", 320, 448),
                    ("free", 448, 12064),
                    ("sum", 12064, 12068),
                ],
            },
            {
                "name": "ReLU",
                "segments": [
                    ("ReLU out", 0, 128),
                    ("free", 128, 320),
                    ("old FC1", 320, 448),
                    ("free", 448, 12068),
                ],
            },
            {
                "name": "FC2 sums",
                "segments": [
                    ("FC2 input", 0, 128),
                    ("free", 128, 12064),
                    ("sum", 12064, 12068),
                ],
            },
            {
                "name": "FC2 matmul",
                "segments": [
                    ("FC2 input", 0, 128),
                    ("logits", 128, 300),
                    ("free", 300, 12064),
                    ("sum", 12064, 12068),
                ],
            },
            {
                "name": "Dequant logits",
                "segments": [
                    ("fp32 logits", 0, 128),
                    ("old int8", 128, 300),
                    ("free", 300, 12068),
                ],
            },
        ],
    }
]
