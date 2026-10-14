import argparse
import json
import os

import torch
from scipy.io.wavfile import write

import CoreAudioML.miscfuncs as miscfuncs
import CoreAudioML.training as training
import CoreAudioML.dataset as dataset
import CoreAudioML.networks as networks


parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True)
parser.add_argument("--data_location", required=True)
parser.add_argument("--file_name", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--metrics", required=True)
parser.add_argument("--test_chunk", type=int, default=100000)
parser.add_argument("--cuda", type=int, default=1)

args = parser.parse_args()


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------

use_cuda = bool(args.cuda) and torch.cuda.is_available()

if use_cuda:
    torch.cuda.empty_cache()
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("Using CUDA")
else:
    print("Using CPU")


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Loading model:")
print(args.model)

model_data = miscfuncs.json_load(args.model)
network = networks.load_model(model_data)

if use_cuda:
    network = network.cuda()

network.eval()


# ---------------------------------------------------------
# Load test dataset
# ---------------------------------------------------------

data = dataset.DataSet(data_dir=args.data_location)

data.create_subset("test")
data.load_file(
    os.path.join("test", args.file_name),
    "test"
)


# ---------------------------------------------------------
# Same losses used during training
# ---------------------------------------------------------

loss_functions = training.LossWrapper(
    {"ESRPre": 0.75, "DC": 0.25},
    [-0.85, 1]
)

loss_esr = training.ESRLoss()


# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

print("Processing test set...")

with torch.no_grad():
    test_output, test_loss = network.process_data(
        data.subsets["test"].data["input"][0],
        data.subsets["test"].data["target"][0],
        loss_functions,
        args.test_chunk
    )

    test_esr = loss_esr(
        test_output,
        data.subsets["test"].data["target"][0]
    )


# ---------------------------------------------------------
# Save audio
# ---------------------------------------------------------

output_cpu = test_output.detach().cpu().numpy()[:, 0, 0]

write(
    args.output,
    data.subsets["test"].fs,
    output_cpu
)


# ---------------------------------------------------------
# Save metrics
# ---------------------------------------------------------

metrics = {
    "test_loss": float(test_loss.item()),
    "test_ESR": float(test_esr.item()),
    "model": args.model,
    "file_name": args.file_name
}

with open(args.metrics, "w") as f:
    json.dump(metrics, f, indent=4)


print()
print("Evaluation complete")
print(f"Test loss: {metrics['test_loss']:.6f}")
print(f"Test ESR:  {metrics['test_ESR']:.6f}")
print()
print("Output:")
print(args.output)
print()
print("Metrics:")
print(args.metrics)