import argparse
import sys
from pathlib import Path

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.dataset import DEFAULT_DATA_ROOT, build_data_loaders
from common.train_utils import build_train_result, get_run_dir, load_checkpoint, save_checkpoint, save_json, save_step_txt_dump
from torch_impl.quantized import QASSGD, project_quantized_parameters
from torch_impl.quantized.models import QuantizedMNISTNet, initialize_quantized_model_from_ptq_checkpoint


TORCH_IMPL_DIR = Path(__file__).resolve().parent
ROOT_DIR = TORCH_IMPL_DIR.parent
DEFAULT_OUTPUT_ROOT = TORCH_IMPL_DIR / "artifacts" / "runs"


def _named_parameter_snapshot(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
    }


def _named_gradient_snapshot(model: nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if param.grad is None else param.grad.detach().cpu().clone()
        for name, param in model.named_parameters()
    }


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    desc: str,
    quantized_mode: bool,
    epoch_index: int,
    global_step_start: int,
    run_dir: Path,
    dump_step_txt: bool,
) -> tuple[dict[str, float], list[dict], int]:
    from tqdm import tqdm

    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    global_step = global_step_start
    step_history = []
    progress = tqdm(data_loader, desc=desc, leave=False)
    for step_in_epoch, (images, labels) in enumerate(progress):
        optimizer.zero_grad()
        weights_before = _named_parameter_snapshot(model) if dump_step_txt else None
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        gradients = _named_gradient_snapshot(model) if dump_step_txt else None
        if hasattr(optimizer, "pre_step"):
            optimizer.pre_step(model)
        optimizer.step()
        if quantized_mode:
            project_quantized_parameters(model)
        weights_after = _named_parameter_snapshot(model) if dump_step_txt else None

        dump_path = None
        if dump_step_txt:
            dump_path = save_step_txt_dump(
                run_dir=run_dir,
                epoch_index=epoch_index,
                step_in_epoch=step_in_epoch,
                global_step=global_step,
                images=images,
                labels=labels,
                logits=logits,
                loss=loss,
                weights_before=weights_before,
                gradients=gradients,
                weights_after=weights_after,
            )

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += images.size(0)
        step_top1 = 100.0 * (logits.argmax(dim=1) == labels).sum().item() / images.size(0)
        step_history.append({
            "epoch": epoch_index,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "loss": loss.item(),
            "top1": step_top1,
            "step_txt_dump": None if dump_path is None else str(dump_path),
        })
        global_step += 1
        progress.set_postfix({"loss": f"{total_loss / total_samples:.4f}", "top1": f"{100.0 * total_correct / total_samples:.2f}"})

    return {"loss": total_loss / total_samples, "top1": 100.0 * total_correct / total_samples}, step_history, global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train quantized / QAS route")
    parser.add_argument("--mode", choices=["quantized", "qas"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--init-from", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--test-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--quantized-lr", type=float, default=0.001)
    parser.add_argument("--qas-lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dump-step-txt", action="store_true")
    return parser.parse_args()


def build_model_and_optimizer(args: argparse.Namespace) -> tuple[nn.Module, torch.optim.Optimizer, float, bool]:
    if args.init_from is None:
        raise ValueError(f"{args.mode} runner expects --init-from PTQ checkpoint")
    checkpoint = load_checkpoint(args.init_from)
    if checkpoint["mode"] != "ptq":
        raise ValueError(f"{args.mode} runner expects PTQ checkpoint, got {checkpoint['mode']}")

    model = QuantizedMNISTNet()
    initialize_quantized_model_from_ptq_checkpoint(model, checkpoint)
    if args.mode == "quantized":
        lr = args.quantized_lr
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum)
    else:
        lr = args.qas_lr
        optimizer = QASSGD(model.parameters(), lr=lr, momentum=args.momentum)
    return model, optimizer, lr, True


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    train_loader, _ = build_data_loaders(
        args.data_root,
        args.batch_size,
        args.test_batch_size,
        args.num_workers,
    )
    model, optimizer, lr, quantized_mode = build_model_and_optimizer(args)
    criterion = nn.CrossEntropyLoss()
    run_dir = get_run_dir(args.output_root, args.run_name)

    print(f"===== Training mode={args.mode}, lr={lr}, run_dir={run_dir} =====")

    epoch_history = []
    step_history = []
    global_step = 0
    for epoch in range(args.epochs):
        train_metrics, epoch_steps, global_step = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            desc=f"{args.mode} train {epoch + 1}/{args.epochs}",
            quantized_mode=quantized_mode,
            epoch_index=epoch,
            global_step_start=global_step,
            run_dir=run_dir,
            dump_step_txt=args.dump_step_txt,
        )
        epoch_history.append({"epoch": epoch, "train": train_metrics})
        step_history.extend(epoch_steps)
        print(f"{args.mode} epoch={epoch} train_top1={train_metrics['top1']:.2f} train_loss={train_metrics['loss']:.4f}")

    init_from_mode = None
    if args.init_from is not None:
        init_from_mode = load_checkpoint(args.init_from)["mode"]
    train_result = build_train_result(args, epoch_history, step_history, lr, init_from_mode)
    save_json(run_dir / "train_result.json", train_result)
    save_checkpoint(run_dir / "checkpoint.pt", {
        "mode": args.mode,
        "model_state_dict": model.state_dict(),
        "train_config": train_result,
    })
    print(f"Saved training artifacts to {run_dir}")


if __name__ == "__main__":
    main()
