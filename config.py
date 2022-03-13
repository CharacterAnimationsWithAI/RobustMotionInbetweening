import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


data = {
    "path": "./lafan1/lafan1/",
    "path_flipped": "./lafan1/lafan1/flipped/",
    "path_small": "./lafan1/lafan1_small/",
    "path_small_flipped": "./lafan1/lafan1_small/flipped/",
    "offsets": [
        [-42.198200, 91.614723, -40.067841],
        [0.103456, 1.857829, 10.548506],
        [43.499992, -0.000038, -0.000002],
        [42.372192, 0.000015, -0.000007],
        [17.299999, -0.000002, 0.000003],
        [0.000000, 0.000000, 0.000000],             # Probably root joint?? or a edge           
        [0.103457, 1.857829, -10.548503],
        [43.500042, -0.000027, 0.000008],
        [42.372257, -0.000008, 0.000014],
        [17.299992, -0.000005, 0.000004],
        [0.000000, 0.000000, 0.000000],             # Probably root joint??
        [6.901968, -2.603733, -0.000001],
        [12.588099, 0.000002, 0.000000],
        [12.343206, 0.000000, -0.000001],
        [25.832886, -0.000004, 0.000003],
        [11.766620, 0.000005, -0.000001],
        [0.000000, 0.000000, 0.000000],             # Probably root joint??
        [19.745899, -1.480370, 6.000108],
        [11.284125, -0.000009, -0.000018],
        [33.000050, 0.000004, 0.000032],
        [25.200008, 0.000015, 0.000008],
        [0.000000, 0.000000, 0.000000],             # Probably root joint??
        [19.746099, -1.480375, -6.000073],
        [11.284138, -0.000015, -0.000012],
        [33.000092, 0.000017, 0.000013],
        [25.199780, 0.000135, 0.000422],
        [0.000000, 0.000000, 0.000000],             # Probably root joint??
    ],
    "parents": [
        -1,
        0,
        1,
        2,
        3,
        4,
        0,
        6,
        7,
        8,
        9,
        0,
        11,
        12,
        13,
        14,
        15,
        13,
        17,
        18,
        19,
        20,
        13,
        22,
        23,
        24,
        25,
    ],
    "joints_to_remove": [5, 10, 16, 21, 26],
    "foot_index": [9, 10, 11, 12, 13, 14, 21, 22, 23, 24, 25, 26],
    "seq_length": 150,
    "offset": 60,
    "num_workers": 0,
}

model = {
    "state_input_dim": 95,
    "offset_input_dim": 91,
    "target_input_dim": 88,
    "lstm_dim": 768,
    "decoder_output_dim": 95,
    "num_joints": 22,
}

train = {
    "batch_size": 128,
    "lr": 0.0001,
    "beta1": 0.5,
    "beta2": 0.9,
    "loss_pos_weight": 1.0,
    "loss_quat_weight": 1.0,
    "loss_root_weight": 0.5,
    "loss_contact_weight": 0.1,
    "loss_slide_weight": 0.1,
    "loss_adv_weight": 0.01,
    "num_epoch": 200,
    "weight_decay": 0.00001,
    "use_ztta": True,
    "use_adv": True,
    "debug": False,
    "method": '',
}

test = {   
    "batch_size": 32,
    "num_epoch": 1,
    "use_ztta": True,
    "use_adv": True,
    "save_img": True,
    "save_gif": True,
    "save_pose": False,
    "save_bvh": False,
    "debug": False,
}