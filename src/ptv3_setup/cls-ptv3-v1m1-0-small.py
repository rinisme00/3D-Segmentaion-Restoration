# ============================================================================ #
# PTv3-Small — Fantastic Breaks Binary Classification
# Classes: 0=complete, 1=broken | 240 train / 60 test
# Input:   coord [N,3] + feat [N,8] → in_channels=11
# ============================================================================ #

_base_ = ["../_base_/default_runtime.py"]

# misc
batch_size = 16
num_worker = 4
batch_size_val = 16
empty_cache = False
enable_amp = True  # save memory on RTX 3090

# ── Model ────────────────────────────────────────────────────────────────────
model = dict(
    type="DefaultClassifier",
    num_classes=2,
    backbone_embed_dim=256,
    backbone=dict(
        type="PT-v3m1",
        in_channels=11,  # coord(3) + geom_features(8) concatenated by Collect
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 2, 2),          # ← smaller than Base
        enc_channels=(32, 64, 128, 128, 256), # ← narrower
        enc_num_head=(2, 4, 8, 8, 16),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 128),
        dec_num_head=(4, 4, 8, 8),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.1,             # lower drop_path for small dataset
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,         # RTX 3090 (Ampere) supports FlashAttention
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=True,             # encoder-only for classification
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("FantasticBreaks",),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
    ],
)

# ── Optimizer & Scheduler ────────────────────────────────────────────────────
epoch = 200
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset_type = "FantasticBreaksClsDataset"
data_root = "data/fantastic-breaks-classification"
class_names = ["complete", "broken"]

data = dict(
    num_classes=2,
    ignore_index=-1,
    names=class_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True),
            dict(type="RandomShift", shift=((-0.2, 0.2), (-0.2, 0.2), (-0.2, 0.2))),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "feat"],  # concat → 3+8 = 11D input
            ),
        ],
        test_mode=False,
        loop=4,  # oversample since dataset is small (240 × 4 = 960 per epoch)
    ),
    val=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
            dict(
                type="GridSample",
                grid_size=0.01,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "category"),
                feat_keys=["coord", "feat"],
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        class_names=class_names,
        transform=[
            dict(type="NormalizeCoord"),
        ],
        test_mode=True,
        test_cfg=dict(
            post_transform=[
                dict(
                    type="GridSample",
                    grid_size=0.01,
                    hash_type="fnv",
                    mode="train",
                    return_grid_coord=True,
                ),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord"),
                    feat_keys=["coord", "feat"],
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[1, 1], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
                [dict(type="RandomScale", scale=[0.8, 1.2], anisotropic=True)],
            ],
        ),
    ),
)

# ── Hooks ────────────────────────────────────────────────────────────────────
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ClsEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# tester
test = dict(type="ClsVotingTester", num_repeat=10)
