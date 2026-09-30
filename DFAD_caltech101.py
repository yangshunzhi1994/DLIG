"""Standalone offline DFAD baseline for Caltech101.

The training update follows the official DFAD Caltech101 implementation:
five student steps followed by one adversarial generator step, using L1 logit
discrepancy.  Dataset and checkpoint handling are kept local to this file.
"""
import argparse
import json
import os
import random
import tempfile

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from datafree.hooks import DeepInversionHook
from datafree.models.ours_generator import HierarchicalGenerator


NUM_CLASSES = 101


class Caltech101Split(Dataset):
    """Prepared official train/test split, excluding BACKGROUND_Google."""

    extensions = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, root, train, transform=None):
        self.root = os.path.abspath(root)
        self.transform = transform
        split = "train" if train else "test"
        split_root = os.path.join(self.root, "101_ObjectCategories_split", split)
        if not os.path.isdir(split_root):
            raise FileNotFoundError("Caltech101 split not found: " + split_root)
        self.classes = sorted(
            name for name in os.listdir(split_root)
            if name != "BACKGROUND_Google"
            and os.path.isdir(os.path.join(split_root, name)))
        if len(self.classes) != NUM_CLASSES:
            raise RuntimeError(
                "Expected 101 object classes after excluding BACKGROUND_Google, "
                "but found %d in %s" % (len(self.classes), split_root))
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples = []
        for class_name in self.classes:
            class_root = os.path.join(split_root, class_name)
            for filename in sorted(os.listdir(class_root)):
                if filename.lower().endswith(self.extensions):
                    self.samples.append((os.path.join(class_root, filename),
                                         self.class_to_idx[class_name]))
        if not self.samples:
            raise RuntimeError("No images found in: " + split_root)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, target


class Flatten(nn.Module):
    def forward(self, tensor):
        return tensor.view(tensor.shape[0], -1)


class GeneratorB(nn.Module):
    """GeneratorB from the official DFAD implementation."""

    def __init__(self, nz=256, ngf=64, nc=3, img_size=128, slope=0.2):
        super().__init__()
        self.init_size = img_size // 16
        self.project = nn.Sequential(
            Flatten(), nn.Linear(nz, ngf * 8 * self.init_size * self.init_size))
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
            if isinstance(module, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
                nn.init.normal_(module.weight, 0.0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 1.0, 0.02)
                nn.init.constant_(module.bias, 0)

    def forward(self, noise):
        output = self.project(noise)
        output = output.view(output.shape[0], -1, self.init_size, self.init_size)
        return self.main(output)


def load_state(model, path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a state_dict: " + path)
    checkpoint = {(key[7:] if key.startswith("module.") else key): value
                  for key, value in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)


def build_resnet(factory, num_classes):
    """Construct a torchvision ResNet across old and new torchvision APIs."""
    try:
        return factory(weights=None, num_classes=num_classes)
    except TypeError:
        return factory(pretrained=False, num_classes=num_classes)


def make_loaders(args, use_cuda):
    test_transform = transforms.Compose([
        transforms.Resize(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    train_set = Caltech101Split(args.data_root, train=True)
    test_set = Caltech101Split(args.data_root, train=False,
                               transform=test_transform)
    if train_set.classes != test_set.classes:
        raise RuntimeError("Caltech101 train/test class order does not match")
    loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=use_cuda)
    print("Caltech101: 101 classes, %d train images, %d test images; "
          "BACKGROUND_Google excluded." % (len(train_set), len(test_set)), flush=True)
    return loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum = 0.0
    correct1 = 0
    correct5 = 0
    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        output = model(data)
        loss_sum += F.cross_entropy(output, target, reduction="sum").item()
        predictions = output.topk(5, dim=1).indices
        correct1 += predictions[:, :1].eq(target[:, None]).sum().item()
        correct5 += predictions.eq(target[:, None]).sum().item()
    count = len(loader.dataset)
    return {"loss": loss_sum / count, "acc1": correct1 / count,
            "acc5": correct5 / count}


def feature_variance_loss(features):
    features = F.adaptive_avg_pool2d(features, 1).flatten(1)
    features = F.normalize(features, dim=1)
    return -features.std(dim=0, unbiased=False).mean()


def synthesis_losses(teacher_logits, fused_features, bn_hooks):
    probabilities = teacher_logits.softmax(dim=1)
    confidence_loss = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(1).mean()
    class_distribution = probabilities.mean(dim=0)
    balance_loss = (class_distribution * class_distribution.clamp_min(1e-8).log()).sum()
    bn_loss = sum((hook.r_feature for hook in bn_hooks),
                  teacher_logits.new_zeros(())) / max(len(bn_hooks), 1)
    diversity_loss = feature_variance_loss(fused_features)
    return bn_loss, confidence_loss, balance_loss, diversity_loss


class GeneratedImageSaver:
    """Keep class-diverse samples ranked by teacher confidence, not realism."""

    def __init__(self, args, classes):
        root = os.path.join(args.save_dir, "generated")
        os.makedirs(root, exist_ok=True)
        self.root = tempfile.mkdtemp(prefix="run-", dir=root)
        self.classes = classes
        self.count = args.save_img_count
        self.min_confidence = args.save_img_min_confidence
        self.candidates = {}
        self.best = {}
        # A separate RNG keeps visualization from changing training noise.
        rng = torch.Generator(device="cpu").manual_seed(args.seed)
        self.fixed_z_c = torch.randn(1, args.nz, generator=rng)
        self.fixed_z_s = torch.randn(1, args.nz, generator=rng)
        print("Generated images: " + self.root, flush=True)

    @torch.no_grad()
    def save_fixed_sample(self, generator, device, epoch):
        was_training = generator.training
        try:
            # Evaluation mode avoids updating BN statistics and supports batch 1.
            generator.eval()
            fake, _, _ = generator(self.fixed_z_c.to(device), self.fixed_z_s.to(device))
            pixels = ((fake[0] + 1) * 127.5).clamp(0, 255).round()
            pixels = pixels.to(torch.uint8).cpu().permute(1, 2, 0).numpy()
            directory = os.path.join(self.root, "fixed-sample")
            os.makedirs(directory, exist_ok=True)
            Image.fromarray(pixels).save(os.path.join(directory, "epoch-%03d.png" % epoch))
            if epoch == 0:
                torch.save({"z_c": self.fixed_z_c, "z_s": self.fixed_z_s},
                           os.path.join(directory, "fixed-noise.pt"))
        finally:
            generator.train(was_training)

    @torch.no_grad()
    def collect(self, fake, teacher_logits, epoch, iteration):
        scores, labels = teacher_logits.detach().softmax(1).max(1)
        scores, labels = scores.cpu().tolist(), labels.cpu().tolist()
        # Store at most one CPU image per predicted class for this epoch.
        for index, (score, label) in enumerate(zip(scores, labels)):
            if not np.isfinite(score) or score < self.min_confidence:
                continue
            previous = self.candidates.get(label)
            if previous is not None and score <= previous["confidence"]:
                continue
            sample = fake[index].detach()
            if not torch.isfinite(sample).all().item():
                continue
            pixels = ((sample + 1) * 127.5).clamp(0, 255).round()
            pixels = pixels.to(torch.uint8).cpu().permute(1, 2, 0).numpy()
            self.candidates[label] = {
                "image": Image.fromarray(pixels), "class_id": label,
                "predicted_class": self.classes[label], "confidence": score,
                "epoch": epoch, "iteration": iteration + 1}

    def save_epoch(self, epoch):
        if not self.candidates:
            print("No generated images passed the confidence filter at epoch %d."
                  % epoch, flush=True)
            return
        epoch_dir = os.path.join(self.root, "epoch-%03d" % epoch)
        best_dir = os.path.join(self.root, "best-per-class")
        os.makedirs(epoch_dir, exist_ok=True)
        os.makedirs(best_dir, exist_ok=True)
        selected = sorted(self.candidates.values(),
                          key=lambda item: item["confidence"], reverse=True)[:self.count]
        records = []
        for rank, item in enumerate(selected, 1):
            filename = "%02d-class-%03d-conf-%.4f.png" % (
                rank, item["class_id"], item["confidence"])
            item["image"].save(os.path.join(epoch_dir, filename))
            records.append({key: value for key, value in item.items() if key != "image"})
            records[-1]["file"] = filename
        width, height = selected[0]["image"].size
        columns = min(3, len(selected))
        grid = Image.new("RGB", (columns * width,
                                ((len(selected) + columns - 1) // columns) * height))
        for index, item in enumerate(selected):
            grid.paste(item["image"], ((index % columns) * width,
                                       (index // columns) * height))
        grid.save(os.path.join(epoch_dir, "preview.png"))
        for label, item in self.candidates.items():
            if label in self.best and item["confidence"] <= self.best[label]["confidence"]:
                continue
            filename = "class-%03d.png" % label
            item["image"].save(os.path.join(best_dir, filename))
            self.best[label] = {key: value for key, value in item.items() if key != "image"}
            self.best[label]["file"] = filename
        for directory, metadata in ((epoch_dir, records),
                                    (best_dir, list(self.best.values()))):
            with open(os.path.join(directory, "metadata.json"), "w", encoding="utf-8") as stream:
                json.dump(metadata, stream, ensure_ascii=False, indent=2)
        self.candidates.clear()
        print("Saved %d selected images and preview to %s" %
              (len(selected), epoch_dir), flush=True)


def train_epoch(args, teacher, student, generator, device, optimizers, epoch,
                bn_hooks, image_saver=None):
    teacher.eval()
    student.train()
    generator.train()
    optimizer_s, optimizer_g = optimizers
    sum_s = 0.0
    sum_g = 0.0
    for iteration in range(args.epoch_itrs):
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
        # Generator training needs gradients from both networks with respect to
        # the synthesized image.  Teacher parameters are frozen in main(), so
        # this builds only the required input-gradient path.
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
        if image_saver is not None:
            image_saver.collect(fake, teacher_logits, epoch, iteration)
        sum_s += loss_s.item()
        sum_g += loss_g.item()
        if iteration % args.log_interval == 0:
            print("Train Epoch: %d [%d/%d] G_Loss: %.6f S_Loss: %.6f" %
                  (epoch, iteration, args.epoch_itrs, loss_g.item(), loss_s.item()),
                  flush=True)
    if image_saver is not None:
        image_saver.save_epoch(epoch)
    return {"student_loss": sum_s / args.epoch_itrs,
            "generator_loss": sum_g / args.epoch_itrs}


def parse_args():
    parser = argparse.ArgumentParser(description="Official-style offline DFAD Caltech101")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--epoch_itrs", type=int, default=50)
    parser.add_argument("--lr_S", type=float, default=0.05)
    parser.add_argument("--lr_G", type=float, default=1e-3)
    parser.add_argument("--data_root", default="data/caltech101")
    parser.add_argument("--ckpt", default="checkpoints1/pretrained/caltech101-resnet34.pt")
    parser.add_argument("--stu_ckpt")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nz", type=int, default=256)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--step_size", type=int, default=100)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--adv", type=float, default=1.0)
    parser.add_argument("--bn", type=float, default=1.0)
    parser.add_argument("--oh", type=float, default=0.5)
    parser.add_argument("--balance", type=float, default=0.1)
    parser.add_argument("--div", type=float, default=0.1)
    parser.add_argument("--save_dir", default="checkpoints1/dfad-caltech101")
    parser.add_argument("--log_tag", default="dfad-caltech101-ep300")
    parser.add_argument("--save_img", dest="save_img", action="store_true",
                        help="Save selected generated images (enabled by default)")
    parser.add_argument("--no_save_img", dest="save_img", action="store_false")
    parser.set_defaults(save_img=True)
    parser.add_argument("--save_img_count", type=int, default=9,
                        help="Maximum images per epoch, at most one per predicted class")
    parser.add_argument("--save_img_min_confidence", type=float, default=0.0,
                        help="Minimum teacher confidence for saving, between 0 and 1")
    args = parser.parse_args()
    if args.save_img_count <= 0:
        parser.error("--save_img_count must be positive")
    if not 0.0 <= args.save_img_min_confidence <= 1.0:
        parser.error("--save_img_min_confidence must be between 0 and 1")
    return args


def main():
    args = parse_args()
    args.data_root = os.path.abspath(args.data_root)
    args.ckpt = os.path.abspath(args.ckpt)
    args.save_dir = os.path.abspath(args.save_dir)
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError("Teacher checkpoint not found: " + args.ckpt)
    os.makedirs(args.save_dir, exist_ok=True)
    print("Offline mode: no pretrained weights will be downloaded.", flush=True)
    print("Teacher checkpoint: " + args.ckpt, flush=True)

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

    loader = make_loaders(args, use_cuda)
    teacher = build_resnet(models.resnet34, NUM_CLASSES)
    student = build_resnet(models.resnet18, NUM_CLASSES)
    generator = HierarchicalGenerator(args, args.img_size)
    load_state(teacher, args.ckpt)
    if args.stu_ckpt:
        load_state(student, os.path.abspath(args.stu_ckpt))
    teacher = teacher.to(device).eval()
    student = student.to(device)
    generator = generator.to(device)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    bn_hooks = [DeepInversionHook(module, 0) for module in teacher.modules()
                if isinstance(module, nn.BatchNorm2d)]

    optimizer_s = torch.optim.SGD(student.parameters(), lr=args.lr_S,
                                  weight_decay=args.weight_decay,
                                  momentum=args.momentum)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
    scheduler_s = torch.optim.lr_scheduler.StepLR(
        optimizer_s, args.step_size, gamma=args.lr_gamma) if args.scheduler else None
    scheduler_g = torch.optim.lr_scheduler.StepLR(
        optimizer_g, args.step_size, gamma=args.lr_gamma) if args.scheduler else None

    metrics_path = os.path.join(args.save_dir, args.log_tag + "-metrics.jsonl")
    best_path = os.path.join(args.save_dir, args.log_tag + "-best.pt")
    last_path = os.path.join(args.save_dir, args.log_tag + "-last.pt")
    with open(metrics_path, "w", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "config", "args": vars(args)}) + "\n")

    teacher_metrics = evaluate(teacher, loader, device)
    print("Teacher baseline: Acc@1=%.4f Acc@5=%.4f Loss=%.4f" %
          (100 * teacher_metrics["acc1"], 100 * teacher_metrics["acc5"],
           teacher_metrics["loss"]), flush=True)
    if teacher_metrics["acc1"] < 0.5:
        raise RuntimeError(
            "Teacher baseline is unexpectedly low; stop before distillation. "
            "Check checkpoint and Caltech101 split/class ordering.")
    with open(metrics_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "teacher_baseline", **teacher_metrics}) + "\n")

    best_acc = 0.0
    image_saver = GeneratedImageSaver(args, loader.dataset.classes) if args.save_img else None
    if image_saver is not None:
        image_saver.save_fixed_sample(generator, device, 0)
    for epoch in range(1, args.epochs + 1):
        losses = train_epoch(args, teacher, student, generator, device,
                             (optimizer_s, optimizer_g), epoch, bn_hooks, image_saver)
        if image_saver is not None:
            image_saver.save_fixed_sample(generator, device, epoch)
        results = evaluate(student, loader, device)
        is_best = results["acc1"] > best_acc
        if is_best:
            best_acc = results["acc1"]
        record = {"epoch": epoch, **losses, **results, "best_acc1": best_acc,
                  "is_best": is_best, "lr_S": optimizer_s.param_groups[0]["lr"],
                  "lr_G": optimizer_g.param_groups[0]["lr"]}
        print("Eval Epoch=%d Acc@1=%.4f Acc@5=%.4f Loss=%.4f Best=%.4f" %
              (epoch, 100 * results["acc1"], 100 * results["acc5"],
               results["loss"], 100 * best_acc), flush=True)
        with open(metrics_path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        checkpoint = {"epoch": epoch, "state_dict": student.state_dict(),
                      "generator": generator.state_dict(), "best_acc1": best_acc,
                      "optimizer_S": optimizer_s.state_dict(),
                      "optimizer_G": optimizer_g.state_dict()}
        torch.save(checkpoint, last_path)
        if is_best:
            torch.save(checkpoint, best_path)
        if args.scheduler:
            scheduler_s.step()
            scheduler_g.step()
    print("Best Acc@1=%.4f" % (100 * best_acc), flush=True)


if __name__ == "__main__":
    main()
