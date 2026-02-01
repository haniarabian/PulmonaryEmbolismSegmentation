import argparse
import random
from models import tiramisu
from dses.MSDatasetInfer import MSDataset
from utils.training import *
from torchvision import transforms as T
from matplotlib import pyplot as plt, cm as cm
from tqdm import tqdm
import os
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import DataLoader

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)

opt_defs = {
    "inference_folder_path": dict(
        flags=("-ifp", "--inference-folder-path"),
        info=dict(
            default=None,
            type=str,
            help="path to folder of images for inference",
            required=True,
        ),
    ),
    "save_path": dict(
        flags=("-sp", "--save-path"),
        info=dict(
            default="./inference_outputs",
            type=str,
            help="path to save output images",
        ),
    ),
    "weights_path": dict(
        flags=("-wp", "--weights-path"),
        info=dict(
            default=None, type=str, help="exact path to model weight", required=True
        ),
    ),
    "fc_num_layers": dict(
        flags=("-fcnl", "--fc-num-layers"),
        info=dict(
            default=67, type=int, help="number of FCDenseNet layers ([57, 67, 103])"
        ),
    ),
    "input_dim": dict(
        flags=("-id", "--input-dim"),
        info=dict(
            default=256,
            type=int,
            help="input dimensions for center cropping"
        ),
    ),
    "bidirectional": dict(
        flags=("-bi", "--bidirectional"),
        info=dict(
            default=False,
            type=bool,
            help="bidirectional c-lstm (use:True, not use: False)",
        ),
    ),
}

parser = argparse.ArgumentParser()
for k, arg in opt_defs.items():
    parser.add_argument(*arg["flags"], **arg["info"])
opt = parser.parse_args()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference_folder_path, save_path, weights_path, fc_num_layers, input_dim, bidirectional = (
        Path(opt.inference_folder_path),
        Path(opt.save_path),
        Path(opt.weights_path),
        opt.fc_num_layers,
        opt.input_dim,
        opt.bidirectional,
    )
    save_path.mkdir(exist_ok=True)
    match fc_num_layers:
        case 57:
            model_constructor = tiramisu.FCDenseNet57
        case 67:
            model_constructor = tiramisu.FCDenseNet67
        case 103:
            model_constructor = tiramisu.FCDenseNet103
    model = model_constructor(
        loss_type="dice",
        n_classes=2,
        grow_rate=12,
        use_stn=False,
        use_sa=True,
        seq_size=3,
        use_lstm=True,
        lstm_kernel_size=3,
        lstm_num_layers=1,
        bidirectional=bidirectional,
    )
    model = model.to(device)
    model_state_dict = torch.load(weights_path, map_location=device , weights_only=False)["model_state"]
    model.load_state_dict(model_state_dict)
    model.eval()

    inference_dataset = MSDataset(
        inference_folder_path,
        transform=T.Compose(
            [
                T.Resize(input_dim),
                T.CenterCrop(input_dim),
                T.Grayscale(3),
                T.ToTensor(),
                T.Normalize([0.3511, 0.3511, 0.3511], [0.2331, 0.2331, 0.2331]),
            ]
        ),
        input_dim=input_dim,
    )

    inference_loader = DataLoader(inference_dataset, batch_size=1)

    for path, data in tqdm(inference_loader):
        inputs = data.cuda()
        outputs = model(inputs)[0]
        indices = range(1, outputs.size(0), 3)
        outputs = outputs[indices, :, :, :]
        preds = get_predictions(outputs)
        #preds = outputs[: , 1 , : , :]
        plt.imsave(
            save_path / path[0].split("/")[-1],
            preds.cpu().detach().numpy()[0],
            format="png",
            cmap="gray",
        )