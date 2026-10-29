"""Check orientation stability of the slide's predicted B and T cells."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".runtime" / "matplotlib"))
sys.path.insert(0, str(ROOT / "script" / "src"))


def main():
    import numpy as np
    import pandas as pd
    import torch
    from model_eval import SAM_ConvNet, get_transforms
    from dataset import CellSamdataset_our
    from runtime import load_classifier_state, select_device
    run = ROOT / "outputs" / "whole-slide-20261012"
    meta = json.loads((run / "run.json").read_text())
    cells = pd.read_csv(meta["slides"][0]["results"])
    cells = cells[cells.celltype.isin(["B", "T"])].reset_index(drop=True)
    _, transform = get_transforms()
    dataset = CellSamdataset_our(cells, transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    device = select_device("mps")
    model = SAM_ConvNet(num_classes=4, pretrained=False).eval()
    model.load_state_dict(load_classifier_state(meta["classifier"]))
    model.to(device)
    predictions = {name: [] for name in ["repeat", "horizontal", "vertical"]}
    with torch.inference_mode():
        for index,(images, _) in enumerate(loader):
            images = images.to(device)
            predictions["repeat"].append(model(images).argmax(1).cpu().numpy())
            predictions["horizontal"].append(model(images.flip([-1])).argmax(1).cpu().numpy())
            predictions["vertical"].append(model(images.flip([-2])).argmax(1).cpu().numpy())
            if index % 20 == 0:
                print(f"Checked {min((index+1)*256,len(cells))}/{len(cells)} B/T predictions", flush=True)
    output = cells[["global_instance_id", "celltype", "prediction", "confidence"]].copy()
    for name, values in predictions.items():
        output[name + "_prediction"] = np.concatenate(values)
    output["stable_both_flips"] = ((output.horizontal_prediction == output.prediction)
                                  & (output.vertical_prediction == output.prediction))
    output.to_csv(run / "tls_analysis" / "all_predicted_BT_flip_checks.csv", index=False)
    print(output.groupby("celltype").stable_both_flips.agg(["size", "sum", "mean"]).to_string())


if __name__ == "__main__":
    main()
