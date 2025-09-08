import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F


from torch.utils.tensorboard import SummaryWriter

from datasets import BatchSampler, collate_fn
from utils.data_utils import (
    file_scanf,
    to_patch,
    x_group,
)
from utils.train_utils import (
    fix_random_seed,
    get_args,
    get_device,
    get_log_dir,
    select_dataset,
    select_model,
)

# 固定随机种子
seed = 1234

fix_random_seed(seed)


args = get_args()
print(args)

# 设备
gpus = [0,1,2]
device = get_device(gpus)

k_fold = KFold(n_splits=5, shuffle=True)

# tensorboard
tb = SummaryWriter(log_dir=get_log_dir())


file_paths = file_scanf()

best_val_accs = []

for fold, (train_idx, val_idx) in enumerate(k_fold.split(file_paths)):

    train_paths = file_paths[train_idx]
    val_paths = file_paths[val_idx]

    train_dataset = select_dataset(train_paths)
    val_dataset = select_dataset(val_paths)
    # 模型
    model = select_model().to(device)
    if fold == 0:
        print(
            "num_params:",
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e4,
            "w",
        )
        print(model)

    train_N = 2   # 2
    val_N = 1

    train_sampler = BatchSampler(train_dataset, args.batch_size, train_N)#########################
    val_sampler = BatchSampler(val_dataset, args.batch_size, val_N)##############

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=lambda batch: collate_fn(batch, train_N),
        num_workers=3,
        prefetch_factor=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=lambda batch: collate_fn(batch, val_N),
        num_workers=1,
        prefetch_factor=2,
        pin_memory=True,
    )

    # 优化器
    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_val_acc = 0
    best_epoch = 0

    for epoch in range(args.epochs):

        # 训练
        model.train()
        total_correct = 0
        total_sample = 0
        total_loss = 0
        for step, (x, y) in enumerate(train_loader):
            total_sample += x.shape[0]
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            # x = to_patch(x, replace=True)
            x = x_group(x)
            y_pred = model(x,y)
            # y_pred = model(x)
            loss_cls = F.cross_entropy(y_pred, y)
            loss = loss_cls
            if step % (len(train_loader) // 5) == 0:
                tb.add_scalar(
                    "fold {}/train/loss_cls".format(fold),
                    loss_cls.item(),
                    epoch * len(train_loader) + step,
                )
            total_loss += loss.item()
            total_correct += (y_pred.argmax(dim=1) == y).sum().item()
            loss.backward()
            optimizer.step()
        train_loss = total_loss / len(train_loader)
        train_acc = total_correct / total_sample * 100
        tb.add_scalar("fold {}/train/loss".format(fold), train_loss, epoch)
        tb.add_scalar("fold {}/train/acc".format(fold), train_acc, epoch)

        # 验证
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_correct = 0
            total_sample = 0
            for x, y in val_loader:
                total_sample += x.shape[0]
                x = x.to(device)
                y = y.to(device)
                # x = to_patch(x)
                x = x_group(x)
                y_pred = model(x)
                loss_cls = F.cross_entropy(y_pred, y)
                total_loss += loss_cls.item()
                total_correct += (y_pred.argmax(dim=1) == y).sum().item()
            val_loss = total_loss / len(val_loader)
            val_acc = total_correct / total_sample * 100
            tb.add_scalar("fold {}/val/loss".format(fold), val_loss, epoch)
            tb.add_scalar("fold {}/val/acc".format(fold), val_acc, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch

            print(
                f"fold {fold}, train_loss {train_loss:.4f}, train_acc {train_acc:.2f}, val_loss {val_loss:.4f}, val_acc {val_acc:.2f}, best_val_acc {best_val_acc:.2f} (epoch {best_epoch}/{epoch})"
            )
    # 将每个fold的最佳验证准确率保存到best_val_accs列表中
    best_val_accs.append(best_val_acc)

# 计算所有fold的最佳验证准确率的平均值和方差
mean_val_acc = sum(best_val_accs) / len(best_val_accs)
var_val_acc = sum((x - mean_val_acc) ** 2 for x in best_val_accs) / len(best_val_accs)

# 记录到TensorBoard
tb.add_scalar("overall/mean_val_acc", mean_val_acc, 0)
tb.add_scalar("overall/var_val_acc", var_val_acc, 0)

# 输出平均值和方差
print(f"Average best validation accuracy: {mean_val_acc:.2f}%")
print(f"Variance of best validation accuracy: {var_val_acc:.2f}")