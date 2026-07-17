#!/usr/bin/env python3
"""Run the Parts 3-6 interpretability suite (misclassifications, feature
maps, saliency, spatial invariance, occlusion) against two already-trained
checkpoints.

Train and save checkpoints first, e.g.:
    python scripts/run_experiment.py --model mlp --save-checkpoint results/checkpoints/mlp.pt
    python scripts/run_experiment.py --model cnn --save-checkpoint results/checkpoints/cnn.pt

Then:
    python scripts/run_interpretability.py \
        --mlp-checkpoint results/checkpoints/mlp.pt \
        --cnn-checkpoint results/checkpoints/cnn.pt
"""
import argparse
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bloodcell.config import DataConfig, TrainConfig, MLPConfig, CNNConfig
from bloodcell.data import BloodCellDataModule
from bloodcell.models import BloodCellMLP, BloodCellCNN
from bloodcell.train import ModelTrainer
from bloodcell.interpret import (
    ErrorAnalyzer, FeatureMapVisualizer, MLPSaliencyVisualizer,
    SpatialInvarianceTester, OcclusionSensitivityAnalyzer, plot_occlusion_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--mlp-checkpoint", required=True)
    parser.add_argument("--cnn-checkpoint", required=True)
    parser.add_argument("--mlp-hidden-dims", nargs="+", type=int, default=[512, 256, 128])
    parser.add_argument("--cnn-conv-channels", nargs="+", type=int, default=[32, 64, 128])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = DataConfig(data_root=args.data_root, num_workers=0)
    dm = BloodCellDataModule(data_config)
    dm.setup()
    _, test_loader = dm.get_loaders()
    num_classes = len(dm.classes)
    input_dim = data_config.image_size * data_config.image_size * 3

    mlp = BloodCellMLP(MLPConfig(input_dim=input_dim, hidden_dims=args.mlp_hidden_dims, num_classes=num_classes))
    mlp.load_state_dict(torch.load(args.mlp_checkpoint, map_location=device))

    cnn = BloodCellCNN(CNNConfig(conv_channels=args.cnn_conv_channels, num_classes=num_classes),
                        input_size=data_config.image_size)
    cnn.load_state_dict(torch.load(args.cnn_checkpoint, map_location=device))

    dummy_cfg = TrainConfig(epochs=0, verbose=False)
    mlp_trainer = ModelTrainer(mlp, test_loader, test_loader, dummy_cfg, device=device)
    cnn_trainer = ModelTrainer(cnn, test_loader, test_loader, dummy_cfg, device=device)

    print("== Part 3: misclassification analysis ==")
    ErrorAnalyzer(mlp_trainer, dm, dm.classes).plot(
        ErrorAnalyzer(mlp_trainer, dm, dm.classes).find_misclassified(test_loader, n=5), "MLP Misclassifications")
    ErrorAnalyzer(cnn_trainer, dm, dm.classes).plot(
        ErrorAnalyzer(cnn_trainer, dm, dm.classes).find_misclassified(test_loader, n=5), "CNN Misclassifications")

    print("== Part 4: CNN feature maps ==")
    random_idx = random.choice(range(len(dm.test_dataset)))
    sample_image, _ = dm.test_dataset[random_idx]
    FeatureMapVisualizer(cnn, device=device).visualize(sample_image, dm.unnormalize)

    print("== Part 5: MLP saliency map ==")
    MLPSaliencyVisualizer(mlp, device=device).visualize(sample_image, dm.unnormalize)

    print("== Part 6: spatial invariance ==")
    SpatialInvarianceTester(mlp_trainer, cnn_trainer, dm, dm.classes).run_and_plot()

    print("== Part 7: occlusion sensitivity ==")
    mlp_occ = OcclusionSensitivityAnalyzer(mlp_trainer, patch_size=8)
    cnn_occ = OcclusionSensitivityAnalyzer(cnn_trainer, patch_size=8)
    tester = SpatialInvarianceTester(mlp_trainer, cnn_trainer, dm, dm.classes)
    samples = tester.find_agreeing_samples()
    for label in sorted(samples.keys()):
        image, true_label = samples[label]
        plot_occlusion_comparison(image, true_label, dm.classes, dm, mlp_occ, cnn_occ)


if __name__ == "__main__":
    main()
