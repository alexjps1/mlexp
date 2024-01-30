"""
Image model for CIFAR-10

Usage:
python3 cifar-model.py <model dir> <cpu ordinal>

The model directory should contain model definition and training
parameters in a file named model_definition.py

The CPU ordinal should be passed in from slurm job assignment
using the $SLURM_JOB_GPUS environment variable.
"""

# import modules
import os
import sys

if len(sys.argv) <= 1:
    print("No model directory provided. Quitting...")
    exit(1)

# resolve importing model_definition.py from subdir
project_root = os.path.abspath(os.path.dirname(__file__))
model_dir = f"{project_root}/{sys.argv[1]}"
if not os.path.isdir(model_dir):
    print("Invalid model directory provided. Quitting...")
    exit(1)
sys.path.insert(0, model_dir)
import model_definition

# import ML modules
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


print("""
-------------
GPU Selection
-------------
""")
# store user's desired GPU, if specified
desired_gpu_ordinal = None
if len(sys.argv) <= 2:
    print("No GPU ordinal provided")
else:
    try:
        desired_gpu_ordinal = int(sys.argv[2])
        print(f"Specified desired GPU ordinal {desired_gpu_ordinal}")
    except ValueError:
        print(f"Invalid desired GPU ordinal {sys.argv[1]} provided")

# printoutput information about available GPUs
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    # show information for each GPU
    print(f"Found {gpu_count} GPU devices. Showing info:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(torch.cuda.get_device_properties(i))
    # use desired GPU if available
    if desired_gpu_ordinal and 0 <= desired_gpu_ordinal < torch.cuda.device_count():
        device = f"cuda:{desired_gpu_ordinal}"
    else:
        print(f"No GPU found with desired GPU ordinal")
        device = "cuda"
    print(f"CUDA Current Device: {torch.cuda.current_device()}")
elif torch.backends.mps.is_available():
    print("CUDA is not available. Using MPS")
    device = "mps"
else:
    print("CUDA and MPS are not available. Using CPU")
    device = "cpu"

# printoutput final device selection
print(f"Model is using device {device}\n")

print("""
----------------------------------------
Model Definition and Training Parameters
----------------------------------------
""")

print(f"Using model and training parameters from {model_dir}/model_definition.py")
MODEL = model_definition.MODEL.to(device)
print(f"Model: {str(MODEL)}")

# set training parameters
LEARNING_RATE = model_definition.LEARNING_RATE
BATCH_SIZE = model_definition.BATCH_SIZE
EPOCHS = model_definition.EPOCHS
CHECKPOINT = model_definition.CHECKPOINT

print(f"Learning rate: {LEARNING_RATE}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Saving state dicts every {CHECKPOINT} epochs")

LOSS_FUNCTION = model_definition.LOSS_FUNCTION
OPTIMIZER = model_definition.OPTIMIZER

print(f"Loss function: {str(LOSS_FUNCTION)}")
print(f"Optimizer: {str(OPTIMIZER)}")

print("""
-------------
Training Data
-------------
""")

# download and define the testing and training data
training_dataset = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)
testing_dataset = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

print("Training data found/downloaded")

# wrap in dataloaders to feed model in batches
training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE)
testing_dataloader = DataLoader(testing_dataset, batch_size=BATCH_SIZE)


print("""
--------------
Load/New Model
--------------
""")

# FIXME functionality to load model using argv
"""
# load saved model
try:
    MODEL.load_state_dict(torch.load(f"{model_dir}/cifar_model_weights.pth", map_location=torch.device(device)))
    print("Loaded saved model weights successfully")
except:
    print("Starting with a blank model")
"""
print("Starting with a blank model")


# define the training process
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # move to GPU
        X = X.to(device)
        y = y.to(device)

        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# define the training process
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            # move to GPU
            X = X.to(device)
            y = y.to(device)

            # make prediction and run loss function
            pred = MODEL(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    result = {"accuracy": correct, "avgloss": test_loss}
    return result

def save_state_dict(model):
    # save the state dict
    state_dict_path = f"{model_dir}/cifar_model_weights.pth"
    torch.save(model.state_dict(), state_dict_path)
    print(f"Saved weights state dict to {state_dict_path}")


def save_whole_model(model):
    # save the whole model
    whole_model_path = f"{model_dir}/cifar_model.pth"
    torch.save(model, whole_model_path)
    print(f"Saved whole model to {whole_model_path}")


print("""
--------------------
Training and Testing
--------------------
""")

# run the train/test loops
record_file = open(f"{model_dir}/records.txt", "w")
for t in range(EPOCHS):
    print(f"Epoch {t + 1}\n-------------------------------")
    train_loop(training_dataloader, MODEL, LOSS_FUNCTION, OPTIMIZER)
    result = test_loop(testing_dataloader, MODEL, LOSS_FUNCTION)
    record_file.write(f"epoch {t}\n")
    record_file.write(f"{str(result)}\n")
    record_file.flush()
    if (t+1) % CHECKPOINT == 0:
        print(f"Reached {t+1} epochs, saving state dict")
        try:
            save_state_dict(MODEL)
        except:
            print("Failed to save state dict")
record_file.close()

print("Done!")

print("""
------------
Saving Model
------------
""")

save_state_dict(MODEL)
save_whole_model(MODEL)
