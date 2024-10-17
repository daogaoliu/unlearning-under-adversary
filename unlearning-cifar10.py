import os
import requests  # type: ignore
import numpy as np  # type: ignore
import argparse
import torch  # type: ignore # type: ignore
from torch import nn  # type: ignore
from torch import optim  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
import tqdm  # type: ignore
import torchvision  # type: ignore
from torchvision import transforms  # type: ignore # type: ignore
from torchvision.models import resnet18  # type: ignore
import pandas as pd  # type: ignore
from torch.utils.data import DataLoader  # type: ignore
import higher  # type: ignore
import copy
import torch.nn.functional as F  # type: ignore

from utils import (  # type: ignore
    normalize_per_image,
    visualize_and_save_tensor,
    calculate_l2_norms,
    CustomImageDataset,
    accuracy,
    get_data,
)

# Disable certain cuDNN features
os.environ["CUDNN_CONVOLUTION_FWD_ALGO"] = "0"
os.environ["CUDNN_CONVOLUTION_BWD_DATA_ALGO"] = "0"
os.environ["CUDNN_CONVOLUTION_BWD_FILTER_ALGO"] = "0"

# Ensure deterministic behavior
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


record_step_optimized = {
    0,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    5000,
}


def generate_normalized_noise_black_box(
    p_size, batch_size, channels, height, width, device
):
    noise = torch.randn(p_size, batch_size, channels, height, width, device=device)
    noise = noise / noise.norm(p=2, dim=(2, 3, 4), keepdim=True)
    return noise


def training_step_white_box(
    args, model, adv_X, adv_y, normal_X, normal_y
):
    adv_X = adv_X.detach().requires_grad_(True)
    if args.unlearn_method == "ga":
        criterion = nn.CrossEntropyLoss()
        model_optimizer = optim.SGD(
            model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=5e-4
        )

        with higher.innerloop_ctx(model, model_optimizer) as (fmodel, diffopt):

            # 1. Hypothetical unlearning
            for _ in range(args.unlearn_epochs):
                unlearn_loss = criterion(fmodel(adv_X), adv_y)
                diffopt.step(-unlearn_loss)

        # 2. Evaluate the loss of the "unlearned model" on normal examples and update adv_X
        performance_loss = criterion(fmodel(normal_X), normal_y)
        grad_of_adv = torch.autograd.grad(-performance_loss, adv_X, allow_unused=True)[
            0
        ]
    elif args.unlearn_method == "ga_gdr":
        criterion = nn.CrossEntropyLoss()
        model_optimizer = optim.SGD(
            model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=5e-4
        )

        with higher.innerloop_ctx(model, model_optimizer) as (fmodel, diffopt):

            for _ in range(args.unlearn_epochs):
                unlearn_loss = criterion(fmodel(adv_X), adv_y)
                retain_loss = criterion(fmodel(normal_X), normal_y)
                total_loss = -unlearn_loss + retain_loss

                diffopt.step(total_loss)
            performance_loss = criterion(fmodel(normal_X), normal_y)
            grad_of_adv = torch.autograd.grad(
                -performance_loss, adv_X, allow_unused=True
            )[0]
    elif args.unlearn_method == "ga_klr":
        ori_model = copy.deepcopy(model)
        criterion = nn.CrossEntropyLoss()
        model_optimizer = optim.SGD(
            model.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=5e-4
        )

        with higher.innerloop_ctx(model, model_optimizer) as (fmodel, diffopt):

            for _ in range(args.unlearn_epochs):
                unlearn_loss = criterion(fmodel(adv_X), adv_y)
                with torch.no_grad():
                    ref_outputs = ori_model(normal_X)
                    ref_probs = F.log_softmax(ref_outputs, dim=-1)
                    ref_probs = ref_probs.view(-1, ref_probs.shape[-1])

                current_outputs = fmodel(normal_X)
                current_probs = F.log_softmax(current_outputs, dim=-1)
                current_probs = current_probs.view(-1, current_probs.shape[-1])

                retain_loss = F.kl_div(
                    current_probs, ref_probs, reduction="batchmean", log_target=True
                )
                total_loss = -unlearn_loss + retain_loss
                diffopt.step(total_loss)

                performance_loss = criterion(fmodel(normal_X), normal_y)
                grad_of_adv = torch.autograd.grad(
                    -performance_loss, adv_X, allow_unused=True
                )[0]
    return grad_of_adv, performance_loss


def training_step_black_box(args, model, retain_loader, forget_loader, top_noises):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)
    m = args.m

    evaluated_noises = []

    for (retain_inputs, retain_targets), (adv_X, adv_y) in zip(
        retain_loader, forget_loader
    ):
        retain_inputs = retain_inputs.to(device)
        retain_targets = retain_targets.to(device)
        adv_X = adv_X.to(device)
        adv_y = adv_y.to(device)

        batch_size, channels, height, width = adv_X.size()

        with torch.no_grad():
            loss_ori = criterion(model(retain_inputs), retain_targets)

        for _, noise in top_noises:
            new_noises = generate_normalized_noise_black_box(
                args.p_size, batch_size, channels, height, width, device=adv_X.device
            )
            for new_noise in new_noises:
                noisy_adv_X_plus = adv_X + noise + args.dis * new_noise
                noisy_adv_X_minus = adv_X + noise - args.dis * new_noise

                fmodel_plus = unlearning_black_box(
                    args,
                    copy.deepcopy(model),
                    [(retain_inputs, retain_targets)],
                    [(noisy_adv_X_plus, adv_y)],
                )

                fmodel_minus = unlearning_black_box(
                    args,
                    copy.deepcopy(model),
                    [(retain_inputs, retain_targets)],
                    [(noisy_adv_X_minus, adv_y)],
                )

                with torch.no_grad():
                    loss_plus = criterion(fmodel_plus(retain_inputs), retain_targets)
                    loss_minus = criterion(fmodel_minus(retain_inputs), retain_targets)

                if loss_plus < loss_ori and loss_minus < loss_ori and args.ensure_decrease:
                    continue

                directional_derivative = loss_plus - loss_minus
                grad_of_adv = directional_derivative * new_noise / args.dis

                del (
                    noisy_adv_X_plus,
                    noisy_adv_X_minus,
                    fmodel_plus,
                    fmodel_minus,
                    loss_plus,
                    loss_minus,
                )

                updated_noise = noise + args.adv_lr * grad_of_adv

                norm = updated_noise.norm(p=2, dim=(1, 2, 3), keepdim=True)
                updated_noise = updated_noise * torch.min(
                    torch.tensor(1.0, device=device), args.clip_norm / norm
                )
                fmodel_test = unlearning_black_box(
                    args,
                    copy.deepcopy(model),
                    [(retain_inputs, retain_targets)],
                    [(adv_X + updated_noise, adv_y)],
                )
                
                with torch.no_grad():
                    loss_test = criterion(fmodel_test(retain_inputs), retain_targets)
                evaluated_noises.append((abs(loss_test.item()), updated_noise))

        evaluated_noises.sort(reverse=False, key=lambda x: x[0])
        candidate_noises = evaluated_noises[:m]

        while len(candidate_noises) < m:
            index = len(candidate_noises) % len(top_noises)
            candidate_noises.append((0, top_noises[index][1]))

        for i in range(len(candidate_noises)):
            norm = candidate_noises[i][1].norm(p=2, dim=(1, 2, 3), keepdim=True)
            candidate_noises[i] = (
                candidate_noises[i][0],
                candidate_noises[i][1]
                * torch.min(torch.tensor(1.0, device=device), args.clip_norm / norm),
            )

    del (
        retain_inputs,
        retain_targets,
        adv_X,
        adv_y,
        top_noises,
        evaluated_noises,
        loss_ori,
    )
    torch.cuda.empty_cache()
    return candidate_noises, candidate_noises[0][0]


def unlearning(args, net, retain, forget, epochs=1):
    """Unlearning by gradient ascent

    Args:
      net : nn.Module.
        model to use as base of unlearning.
      retain : torch.utils.data.DataLoader.
        Dataset loader for access to the retain set. This is the subset
        of the training set that we don't want to forget.
      forget : torch.utils.data.DataLoader.
        Dataset loader for access to the forget set. This is the subset
        of the training set that we want to forget. This method doesn't
        make use of the forget set.
      validation : torch.utils.data.DataLoader.
        Dataset loader for access to the validation set. This method doesn't
        make use of the validation set.
    Returns:
      net : updated model
    """

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    net.train()

    if args.unlearn_method == "ga":
        for _ in tqdm.tqdm(range(epochs)):
            for inputs, targets in forget:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                (-loss).backward()
                optimizer.step()
            scheduler.step()

    elif args.unlearn_method == "ga_gdr":
        for _ in tqdm.tqdm(range(epochs)):
            for (inputs, targets), (retain_inputs, retain_targets) in zip(
                forget, retain
            ):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                retain_inputs, retain_targets = retain_inputs.to(
                    DEVICE
                ), retain_targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(inputs)
                forget_loss = criterion(outputs, targets)
                retain_loss = criterion(net(retain_inputs), retain_targets)
                total_loss = -forget_loss + retain_loss
                (total_loss).backward()
                optimizer.step()
            scheduler.step()
    elif args.unlearn_method == "ga_klr":
        ori_model = copy.deepcopy(net)
        for _ in tqdm.tqdm(range(epochs)):
            for (inputs, targets), (retain_inputs, retain_targets) in zip(
                forget, retain
            ):
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                retain_inputs, retain_targets = retain_inputs.to(
                    DEVICE
                ), retain_targets.to(DEVICE)
                optimizer.zero_grad()
                outputs = net(inputs)
                forget_loss = criterion(outputs, targets)
                with torch.no_grad():
                    ref_outputs = ori_model(retain_inputs)
                    ref_probs = F.log_softmax(ref_outputs, dim=-1)
                    ref_probs = ref_probs.view(-1, ref_probs.shape[-1])

                    current_outputs = net(retain_inputs)
                    current_probs = F.log_softmax(current_outputs, dim=-1)
                    current_probs = current_probs.view(-1, current_probs.shape[-1])

                    retain_kl_loss = F.kl_div(
                        current_probs, ref_probs, reduction="batchmean", log_target=True
                    )
                total_loss = -forget_loss + retain_kl_loss
                (total_loss).backward()
                optimizer.step()
            scheduler.step()
    net.eval()
    return net


def unlearning_black_box(args, net, retain_loader, forget_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Clone the model to avoid modifying the input model
    net_clone = copy.deepcopy(net)
    net_clone.to(device)  # Ensure the cloned model is on the correct device.

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        net_clone.parameters(), lr=args.unlearn_lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.unlearn_epochs
    )

    net_clone.train()  # Set the model to training mode.

    for _ in range(args.unlearn_epochs):
        for (retain_inputs, retain_targets), (forget_inputs, forget_targets) in zip(
            retain_loader, forget_loader
        ):
            retain_inputs, retain_targets = retain_inputs.to(device), retain_targets.to(
                device
            )
            forget_inputs, forget_targets = forget_inputs.to(device), forget_targets.to(
                device
            )

            optimizer.zero_grad()
            forget_outputs = net_clone(forget_inputs)

            # Compute the losses for both sets
            forget_loss = criterion(forget_outputs, forget_targets)

            (-forget_loss).backward(
                retain_graph=True
            )  # Perform backward to maximize forget_loss and minimize retain_loss

            optimizer.step()  # Update model parameters

        scheduler.step()  # Adjust the learning rate after each epoch

    net_clone.eval()  # Set the model to evaluation mode after training is complete.
    del net
    torch.cuda.empty_cache()
    return net_clone

def evaluation(
    retain_loader,
    normal_loader,
    test_loader,
    adv_X_ori,
    adv_X,
    adv_y,
    step,
    log_folder,
):  
    # if args.transfer_model:
    #     local_path = "resnet18_cifar10_transfer1.pth"
    # else:
    local_path = "weights_resnet18_cifar10.pth"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth"
        )
        open(local_path, "wb").write(response.content)

    weights_pretrained = torch.load(local_path, map_location=DEVICE)
    ft_model = resnet18(weights=None, num_classes=10)
    ft_model.load_state_dict(weights_pretrained)
    ft_model.to(DEVICE)

    forget_dataset = CustomImageDataset(adv_X, adv_y)
    forget_loader = DataLoader(forget_dataset, batch_size=128, shuffle=True)
    if args.save_images:
        visualize_and_save_tensor(
            adv_X,
            root=f"optimized_images/N={args.unlearn_n}",
            filename=f"t={step}_lr={args.adv_lr}_ume={args.unlearn_method}_norm={args.clip_norm}",
        )

    ft_model = unlearning(
        args,
        ft_model,
        retain_loader,
        forget_loader,
        epochs=args.unlearn_epochs,
    )

    retain_acc = accuracy(ft_model, retain_loader, DEVICE)
    normal_acc = accuracy(ft_model, normal_loader, DEVICE)
    forget_acc = accuracy(ft_model, forget_loader, DEVICE)
    test_acc = accuracy(ft_model, test_loader, DEVICE)

    if args.optimize_images:
        l2_norm = calculate_l2_norms(adv_X_ori, adv_X)
    else:
        l2_norm = -1

    res = {
        "unlearn_n": args.unlearn_n,
        "forget_set_seed": args.forget_set_seed,
        "unlearn_epochs": args.unlearn_epochs,
        "unlearn_lr": args.unlearn_lr,
        "adv_lr": args.adv_lr,
        "optimize_images": args.optimize_images,
        "attack_method": args.attack_method,
        "unlearn_method": args.unlearn_method,
        "step_optimized": step,
        "m": args.m,
        "p_size": args.p_size,
        "dis": args.dis,
        "l2_norm": l2_norm,
        "clip_norm": args.clip_norm,
        "retain_acc": retain_acc,
        "normal_acc": normal_acc,
        "forget_acc": forget_acc,
        "test_acc": test_acc,
    }
    if args.attack_method == "white_box":
        if args.optimize_images:
                output_file = "log_white_box.csv"
        else:
            output_file = "log_baseline.csv"
    elif args.attack_method == "black_box":
        output_file = "log_black_box.csv"
    os.makedirs(log_folder, exist_ok=True)

    output_file = os.path.join(log_folder, output_file)
    if os.path.exists(output_file):
        data = pd.read_csv(output_file, index_col=0).reset_index(drop=True)
        data_new = pd.concat([data, pd.DataFrame([res])])
        data_new.to_csv(output_file)
    else:
        data_new = pd.DataFrame([res])
        data_new.to_csv(output_file)


def main(args):

    # manual random seed is used for dataset partitioning to ensure reproducible results across runs
    RNG = torch.Generator().manual_seed(42)

    # download and pre-process CIFAR10
    normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=normalize
    )

    # we split held out data into test and validation set
    held_out = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=normalize
    )
    test_set, val_set = torch.utils.data.random_split(
        held_out, [0.5, 0.5], generator=RNG
    )
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)

    # download the forget and retain index split

    local_path = "forget_idx.npy"

    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/" + local_path
        )
        open(local_path, "wb").write(response.content)
    if args.forget_set_seed < 0:
        forget_idx = np.load(local_path)[: args.unlearn_n]
    else:
        ori_forget_idx = np.load(local_path)
        np.random.seed(args.forget_set_seed)
        forget_idx = np.random.permutation(ori_forget_idx)[: args.unlearn_n]

    # construct indices of retain from those of the forget set
    forget_mask = np.zeros(len(train_set.targets), dtype=bool)
    forget_mask[forget_idx] = True
    retain_idx = np.arange(forget_mask.size)[~forget_mask]

    # split train set into a forget and a retain set

    retain_set = torch.utils.data.Subset(train_set, retain_idx)
    
    retain_loader = torch.utils.data.DataLoader(
        retain_set, batch_size=128, shuffle=True, num_workers=2, generator=RNG
    )
    forget_set = torch.utils.data.Subset(train_set, forget_idx)

    normal_X, normal_y = get_data(retain_set, device=DEVICE)

    normal_dataset = CustomImageDataset(normal_X, normal_y)
    normal_loader = DataLoader(normal_dataset, batch_size=128, shuffle=True)
    # download pre-trained weights
    local_path = "weights_resnet18_cifar10.pth"
    if not os.path.exists(local_path):
        response = requests.get(
            "https://storage.googleapis.com/unlearning-challenge/weights_resnet18_cifar10.pth"
        )
        open(local_path, "wb").write(response.content)
    

    weights_pretrained = torch.load(local_path, map_location=DEVICE)

    ft_model = resnet18(weights=None, num_classes=10)
    ft_model.load_state_dict(weights_pretrained)
    ft_model.to(DEVICE)

    if args.optimize_images:
        torch.manual_seed(42)
        adv_X, adv_y = get_data(forget_set, args.unlearn_n, device=DEVICE)
        adv_X_ori = copy.deepcopy(adv_X)
        visualize_and_save_tensor(
            adv_X, root=f"optimized_images/N={args.unlearn_n}", filename="original"
        )

        if args.step_optimized > 0:
            adv_X.requires_grad = True


    if args.optimize_images and args.step_optimized > 0:
        if args.attack_method == "white_box":
            loss_history = []
            noise = torch.zeros_like(adv_X)
            for step in tqdm.tqdm(range(args.step_optimized)):
                if step in record_step_optimized:
                    evaluation(
                        retain_loader,
                        normal_loader,
                        test_loader,
                        adv_X_ori,
                        adv_X_ori + noise,
                        adv_y,
                        step,
                        args.log_folder,
                    )
                
                grad_of_adv, performance_loss = training_step_white_box(
                    args,
                    copy.deepcopy(ft_model),
                    adv_X_ori + noise,
                    adv_y,
                    normal_X,
                    normal_y,
                )
                with torch.no_grad():
                    noise -= args.adv_lr * grad_of_adv

                print(f"Step {step}, Loss: {performance_loss:.4f}")
                loss_history.append(performance_loss.cpu().detach().numpy().item())
                noise_norm = torch.norm(noise, p=2, dim=(1, 2, 3)).mean().item()
                if noise_norm > args.clip_norm:
                    noise = noise * (
                        args.clip_norm / noise_norm
                    )  # Scale the noise to have the norm equal to clip_norm
                

        elif args.attack_method == "black_box":
            if args.unlearn_method != "ga":
                raise ValueError(
                    f"Error: The unlearning_method '{args.unlearn_method}' is not valid. Please use 'ga' for black-box."
                )

            noise = (
                generate_normalized_noise_black_box(
                    1,
                    adv_X.size(0),
                    adv_X.size(1),
                    adv_X.size(2),
                    adv_X.size(3),
                    device=adv_X.device,
                ).detach()
                * args.clip_norm
            )

            top_noises = [(0, noise[0])]
            for step in tqdm.tqdm(range(args.step_optimized)):
                top_noises, performance_loss = training_step_black_box(
                    args, ft_model, retain_loader, [(adv_X, adv_y)], top_noises
                )
                print(f"Step {step}, Loss: {performance_loss:.4f}")

                if step in record_step_optimized:
                    smallest_acc = float("inf")
                    best_noise = torch.zeros_like(
                        top_noises[0][1], device=DEVICE, requires_grad=True
                    )

                    for _, noise in top_noises:
                        noisy_adv_X = adv_X + noise
                        fmodel_test = unlearning_black_box(
                            args, ft_model, retain_loader, [(noisy_adv_X, adv_y)]
                        )
                        retain_acc = accuracy(fmodel_test, retain_loader, DEVICE)

                    if retain_acc < smallest_acc:
                        smallest_acc = retain_acc
                        best_noise = noise
                        
                    evaluation(
                        retain_loader,
                        normal_loader,
                        test_loader,
                        adv_X_ori,
                        adv_X_ori + best_noise,
                        adv_y,
                        step,
                        args.log_folder,
                    )
        else:
            raise ValueError(f"no such attack method")

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_LABELS = 10
    print("Running on device:", DEVICE.upper())

    parser = argparse.ArgumentParser()
    parser.add_argument("--unlearn_n", type=int, default=100)
    parser.add_argument("--forget_set_seed", type=int, default=-1)
    parser.add_argument("--unlearn_epochs", type=int, default=1)
    parser.add_argument("--unlearn_lr", type=float, default=0.02)
    parser.add_argument("--adv_lr", type=float, default=0.05)

    parser.add_argument("--optimize_images", action="store_true")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--step_optimized", type=int, default=5)
    parser.add_argument("--clip_norm", type=float, default=3)

    parser.add_argument("--unlearn_method", type=str, default="ga")
    parser.add_argument("--log_folder", type=str, default="./log")
    
    # parameters for balck-box attack
    parser.add_argument("--attack_method", type=str, default="white_box")
    parser.add_argument("--p_size", type=int, default=1)
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--dis", type=float, default=0.1)
    parser.add_argument("--ensure_decrease", action="store_true")

    args = parser.parse_args()
    main(args)
