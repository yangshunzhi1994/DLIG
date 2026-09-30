"""Faithful offline baseline of the official DFAD NYUv2 experiment."""
import argparse
import json
import os
import random

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from datafree.datasets import NYUv2
from datafree.models.deeplab import deeplabv3_mobilenet, deeplabv3_resnet50
from datafree.models.ours_generator import HierarchicalGenerator
from datafree.hooks import DeepInversionHook
from datafree.criterions import kldiv


NUM_CLASSES = 13


class NYUTestTransform:
    def __init__(self, size=256):
        self.size = size

    def __call__(self, image, target):
        image = TF.resize(image.convert("RGB"), self.size, InterpolationMode.BILINEAR)
        target = TF.resize(target, self.size, InterpolationMode.NEAREST)
        image = TF.normalize(TF.to_tensor(image), (0.5,) * 3, (0.5,) * 3)
        target = torch.as_tensor(np.asarray(target, dtype=np.int64).copy())
        return image, target


class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.shape[0], -1)


class GeneratorB(nn.Module):
    """Exact GeneratorB architecture from the official DFAD repository."""

    def __init__(self, nz=256, ngf=64, nc=3, img_size=128, slope=0.2):
        super().__init__()
        self.init_size = (img_size // 16, img_size // 16)
        self.project = nn.Sequential(
            Flatten(), nn.Linear(nz, ngf * 8 * self.init_size[0] * self.init_size[1]))
        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf * 8),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4), nn.LeakyReLU(slope, inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2), nn.LeakyReLU(slope, inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.LeakyReLU(slope, inplace=True),
            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf), nn.LeakyReLU(slope, inplace=True),
            nn.Conv2d(ngf, nc, 3, 1, 1), nn.Tanh())
        for module in self.modules():
            if isinstance(module, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.constant_(module.bias, 0)

    def forward(self, noise):
        projected = self.project(noise)
        projected = projected.view(projected.shape[0], -1,
                                   self.init_size[0], self.init_size[1])
        return self.main(projected)


def validate_dataset(root):
    required = ("splits.mat", "image/test", "seg13/test")
    missing = [item for item in required if not os.path.exists(os.path.join(root, item))]
    if missing:
        raise FileNotFoundError("NYUv2 is not prepared; missing: " + ", ".join(missing))


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for name in ("state_dict", "model_state_dict", "model"):
            if name in checkpoint and isinstance(checkpoint[name], dict):
                checkpoint = checkpoint[name]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state_dict")
    return {(key[7:] if key.startswith("module.") else key): value
            for key, value in checkpoint.items()}


def load_model_state(model, path):
    state = unwrap_state_dict(torch.load(path, map_location="cpu"))
    if "classifier.0.convs.0.0.weight" in state and \
            "classifier.classifier.0.convs.0.0.weight" in model.state_dict():
        state = {("classifier.classifier." + key[len("classifier."):]
                  if key.startswith("classifier.") else key): value
                 for key, value in state.items()}
    model.load_state_dict(state, strict=True)


def feature_variance_loss(features):
    features = F.adaptive_avg_pool2d(features, 1).flatten(1)
    features = F.normalize(features, dim=1)
    return -features.std(dim=0, unbiased=False).mean()


def synthesis_losses(teacher_logits, fused_features, bn_hooks):
    probabilities = teacher_logits.softmax(dim=1)
    confidence_loss = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1).mean()
    class_distribution = probabilities.mean(dim=(0, 2, 3))
    balance_loss = (class_distribution * class_distribution.clamp_min(1e-8).log()).sum()
    bn_loss = sum((hook.r_feature for hook in bn_hooks),
                  teacher_logits.new_zeros(())) / max(len(bn_hooks), 1)
    diversity_loss = feature_variance_loss(fused_features)
    return bn_loss, confidence_loss, balance_loss, diversity_loss


def generate_balanced_targets(batch_size, device):
    labels = torch.arange(batch_size, device=device) % NUM_CLASSES
    labels = labels[torch.randperm(batch_size, device=device)]
    return F.one_hot(labels, num_classes=NUM_CLASSES).float()


def custom_cross_entropy(logits, targets):
    targets = targets[:, :, None, None]
    return -(targets.detach() * logits.log_softmax(dim=1)).sum(1).mean()


def save_training_progress(args, student, generator, optimizer_s, optimizer_g,
                           epoch, iteration, fake):
    progress_path = os.path.join(args.save_dir, args.log_tag + "-progress.pt")
    torch.save({"epoch": epoch, "iteration": iteration,
                "state_dict": student.state_dict(),
                "generator": generator.state_dict(),
                "optimizer_S": optimizer_s.state_dict(),
                "optimizer_G": optimizer_g.state_dict()}, progress_path)
    if args.save_img:
        image_dir = os.path.join(args.save_dir, "generated-progress")
        os.makedirs(image_dir, exist_ok=True)
        images = (((fake[:min(9, fake.shape[0])] + 1) / 2) * 255).clamp(0, 255)
        images = images.detach().cpu().numpy().transpose(0, 2, 3, 1).astype("uint8")
        for index, image in enumerate(images):
            Image.fromarray(image).save(os.path.join(
                image_dir, "epoch-%03d-iter-%03d-%02d.png" %
                (epoch, iteration + 1, index)))


def update_confusion(confusion, prediction, target):
    valid = (target >= 0) & (target < NUM_CLASSES)
    encoded = NUM_CLASSES * target[valid] + prediction[valid]
    confusion += torch.bincount(encoded.cpu(), minlength=NUM_CLASSES ** 2).reshape(
        NUM_CLASSES, NUM_CLASSES)


def confusion_results(confusion):
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    class_iou = intersection / union.clamp_min(1)
    return {"pixel_acc": (intersection.sum() / confusion.sum().clamp_min(1)).item(),
            "mean_iou": class_iou.mean().item(), "class_iou": class_iou.tolist()}


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float64)
    for data, target in loader:
        output = model(data.to(device, non_blocking=True))
        update_confusion(confusion, output.argmax(1), target.to(device, non_blocking=True))
    return confusion_results(confusion)


def train(args, teacher, student, generator, device, optimizer, epoch, bn_hooks):
    """DFAD update order with the project's hierarchical synthesis method."""
    teacher.eval()
    student.train()
    generator.train()
    optimizer_s, optimizer_g = optimizer
    mean_student_loss = 0.0
    mean_generator_loss = 0.0
    for iteration in range(args.epoch_itrs):
        # Keep the proven DFAD update ratio.  Fresh noise for every student
        # step is important for dense prediction; repeatedly fitting one
        # selected image quickly reduces synthetic-data diversity.
        for _ in range(5):
            z_c = torch.randn(args.batch_size, args.nz, device=device)
            z_s = torch.randn(args.batch_size, args.nz, device=device)
            optimizer_s.zero_grad(set_to_none=True)
            fake, _, _ = generator(z_c, z_s)
            fake = fake.detach()
            with torch.no_grad():
                teacher_logits = teacher(fake)
            student_logits = student(fake)
            loss_s = F.l1_loss(student_logits, teacher_logits)
            loss_s.backward()
            optimizer_s.step()

        z_c = torch.randn(args.batch_size, args.nz, device=device)
        z_s = torch.randn(args.batch_size, args.nz, device=device)
        optimizer_g.zero_grad(set_to_none=True)
        fake, _, fused_features = generator(z_c, z_s)
        teacher_logits = teacher(fake)
        student_logits = student(fake)
        loss_adv = -torch.log(F.l1_loss(student_logits, teacher_logits) + 1.0)
        loss_bn, loss_confidence, loss_balance, loss_diversity = synthesis_losses(
            teacher_logits, fused_features, bn_hooks)
        loss_g = (args.adv * loss_adv + args.bn * loss_bn +
                  args.oh * loss_confidence + args.balance * loss_balance +
                  args.div * loss_diversity)
        loss_g.backward()
        optimizer_g.step()
        mean_student_loss += loss_s.item()
        mean_generator_loss += loss_g.item()

        if iteration % args.log_interval == 0:
            print("Train Epoch: %d [%d/%d] G: %.6f S: %.6f Adv: %.6f BN: %.6f Conf: %.6f Bal: %.6f Div: %.6f"
                  % (epoch, iteration, args.epoch_itrs, loss_g.item(),
                     loss_s.item(), loss_adv.item(), loss_bn.item(),
                     loss_confidence.item(), loss_balance.item(),
                     loss_diversity.item()), flush=True)
            save_training_progress(args, student, generator, optimizer_s, optimizer_g,
                                   epoch, iteration, fake)

    return {"student_loss": mean_student_loss / args.epoch_itrs,
            "generator_loss": mean_generator_loss / args.epoch_itrs}


@torch.no_grad()
def test(args, student, teacher, generator, device, test_loader):
    student.eval()
    generator.eval()
    teacher.eval()
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.float64)
    image_index = 0
    result_dir = os.path.join(args.save_dir, "nyu-DFAD")
    if args.save_img:
        os.makedirs(result_dir, exist_ok=True)

    for data, target in test_loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        z_c = torch.randn(data.shape[0], args.nz, device=device, dtype=data.dtype)
        z_s = torch.randn(data.shape[0], args.nz, device=device, dtype=data.dtype)
        fake, _, _ = generator(z_c, z_s)
        output = student(data)

        if args.save_img:
            teacher_output = teacher(data)
            input_images = (((data + 1) / 2) * 255).clamp(0, 255).cpu().numpy()
            input_images = input_images.transpose(0, 2, 3, 1).astype("uint8")
            predictions = NYUv2.decode_fn(output.argmax(1).cpu().numpy()).astype("uint8")
            teacher_predictions = NYUv2.decode_fn(
                teacher_output.argmax(1).cpu().numpy()).astype("uint8")
            targets = NYUv2.decode_fn(target.cpu().numpy()).astype("uint8")
            for prediction, image, colored_target, teacher_prediction in zip(
                    predictions, input_images, targets, teacher_predictions):
                Image.fromarray(prediction).save(os.path.join(result_dir, "%d_pred.png" % image_index))
                Image.fromarray(image).save(os.path.join(result_dir, "%d_img.png" % image_index))
                Image.fromarray(colored_target).save(os.path.join(result_dir, "%d_target.png" % image_index))
                Image.fromarray(teacher_prediction).save(
                    os.path.join(result_dir, "%d_teacher.png" % image_index))
                image_index += 1
        update_confusion(confusion, output.argmax(1), target)

    results = confusion_results(confusion)
    print("\nTest set: Acc= %.6f, mIoU: %.6f\n"
          % (results["pixel_acc"], results["mean_iou"]), flush=True)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Official DFAD NYUv2 offline baseline")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=320)
    parser.add_argument("--epoch_itrs", type=int, default=50)
    parser.add_argument("--lr_S", type=float, default=0.1)
    parser.add_argument("--lr_G", type=float, default=1e-3)
    parser.add_argument("--data_root", default="data/NYUv2")
    parser.add_argument("--ckpt", default="checkpoints1/pretrained/nyuv2-deeplabv3_resnet50.pt")
    parser.add_argument("--stu_ckpt")
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--test_size", type=int, default=256)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--lr_gamma", type=float, default=0.3)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_img", action="store_true")
    parser.add_argument("--save_dir", default="checkpoints1/dfad-nyuv2-baseline")
    parser.add_argument("--log_tag", default="dfad-nyuv2-ep320")
    parser.add_argument("--adv", type=float, default=0.5)
    parser.add_argument("--bn", type=float, default=1.0)
    parser.add_argument("--oh", type=float, default=0.5)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--balance", type=float, default=0.1)
    parser.add_argument("--div", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--g_life", type=int, default=10)
    parser.add_argument("--g_loops", type=int, default=2)
    parser.add_argument("--gwp_loops", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    args.data_root = os.path.abspath(args.data_root)
    args.ckpt = os.path.abspath(args.ckpt)
    args.save_dir = os.path.abspath(args.save_dir)
    validate_dataset(args.data_root)
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError("Teacher checkpoint not found: " + args.ckpt)
    os.makedirs(args.save_dir, exist_ok=True)
    print("Offline mode: no pretrained weights will be downloaded.", flush=True)
    print("Teacher checkpoint: %s" % args.ckpt, flush=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda:%d" % args.gpu if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dataset = NYUv2(args.data_root, split="test", num_classes=NUM_CLASSES,
                    transforms=NYUTestTransform(args.test_size))
    loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=use_cuda)
    teacher = deeplabv3_resnet50(num_classes=NUM_CLASSES, pretrained_backbone=False)
    student = deeplabv3_mobilenet(num_classes=NUM_CLASSES, pretrained_backbone=False)
    for module in student.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.5
    generator = HierarchicalGenerator(args, args.img_size)
    load_model_state(teacher, args.ckpt)
    if args.stu_ckpt:
        load_model_state(student, args.stu_ckpt)
    teacher = teacher.to(device).eval()
    student = student.to(device)
    generator = generator.to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    bn_hooks = [DeepInversionHook(module, 0) for module in teacher.modules()
                if isinstance(module, nn.BatchNorm2d)]

    optimizer_s = torch.optim.SGD(student.parameters(), lr=args.lr_S,
                                  weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
    scheduler_s = torch.optim.lr_scheduler.StepLR(
        optimizer_s, args.step_size, gamma=args.lr_gamma) if args.scheduler else None
    scheduler_g = torch.optim.lr_scheduler.StepLR(
        optimizer_g, args.step_size, gamma=args.lr_gamma) if args.scheduler else None

    best_miou = 0.0
    best_epoch = 0
    best_path = os.path.join(args.save_dir, args.log_tag + "-best.pt")
    last_path = os.path.join(args.save_dir, args.log_tag + "-last.pt")
    metrics_path = os.path.join(args.save_dir, args.log_tag + "-metrics.jsonl")
    with open(metrics_path, "w", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "config", "args": vars(args)}) + "\n")
    initial_path = os.path.join(args.save_dir, args.log_tag + "-initial.pt")
    torch.save({"epoch": 0, "iteration": 0, "state_dict": student.state_dict(),
                "generator": generator.state_dict(),
                "optimizer_S": optimizer_s.state_dict(),
                "optimizer_G": optimizer_g.state_dict()}, initial_path)
    print("Initial result saved to %s" % initial_path, flush=True)

    print("Evaluating teacher baseline on %d test images..." % len(dataset), flush=True)
    teacher_metrics = evaluate_model(teacher, loader, device)
    print("Teacher baseline: Acc=%.6f, mIoU=%.6f"
          % (teacher_metrics["pixel_acc"], teacher_metrics["mean_iou"]), flush=True)
    with open(metrics_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "teacher_baseline", **teacher_metrics}) + "\n")

    for epoch in range(1, args.epochs + 1):
        losses = train(args, teacher, student, generator, device,
                       (optimizer_s, optimizer_g), epoch, bn_hooks)
        results = test(args, student, teacher, generator, device, loader)
        is_best = results["mean_iou"] > best_miou
        if is_best:
            best_miou = results["mean_iou"]
            best_epoch = epoch
        record = {"epoch": epoch, **losses, **results, "best_miou": best_miou,
                  "best_epoch": best_epoch, "is_best": is_best,
                  "lr_S": optimizer_s.param_groups[0]["lr"],
                  "lr_G": optimizer_g.param_groups[0]["lr"]}
        print(json.dumps(record), flush=True)
        with open(metrics_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        checkpoint = {"epoch": epoch, "state_dict": student.state_dict(),
                      "generator": generator.state_dict(), "best_miou": best_miou,
                      "best_epoch": best_epoch, "optimizer_S": optimizer_s.state_dict(),
                      "optimizer_G": optimizer_g.state_dict()}
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, best_path)
        if args.scheduler:
            scheduler_s.step()
            scheduler_g.step()

    print("Best mIoU=%.6f at epoch=%d" % (best_miou, best_epoch), flush=True)


if __name__ == "__main__":
    main()
