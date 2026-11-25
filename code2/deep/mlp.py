"""多层感知机（MLP）—— PyTorch 实现

任务：MNIST 手写数字 10 分类
结构：784 -> 256 -> 128 -> 10（ReLU + Dropout）
训练：Adam，交叉熵，3 个 epoch，batch 128

运行：
    python mlp.py
首次运行会自动下载 MNIST 到 ./data。
"""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

SEED = 0
BATCH_SIZE = 128
EPOCHS = 3
LR = 1e-3
DROP_OUT = 0.2


class MLP(nn.Module):
    """三层感知机：两个隐藏层 + 一个输出层。"""

    def __init__(self, in_dim=784, hidden=(256, 128), n_class=10, p_drop=DROP_OUT):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(p_drop)]
            prev = h
        layers.append(nn.Linear(prev, n_class))  # 输出层不加激活，交给 CrossEntropyLoss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))  # [N,1,28,28] -> [N,784]


def build_loaders(root="./data"):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST 的全局均值/标准差
    ])
    train_ds = datasets.MNIST(root, train=True, download=True, transform=tf)
    test_ds = datasets.MNIST(root, train=False, download=True, transform=tf)
    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(test_ds, batch_size=512),
        len(train_ds),
        len(test_ds),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(dim=1) == y).sum().item()
    return correct / len(loader.dataset)


def main():
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dl, test_dl, n_train, n_test = build_loaders()

    model = MLP().to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"device = {device}")
    print(f"params = {n_param:,}")
    print(f"train = {n_train}  test = {n_test}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)

        train_loss = running / n_train
        test_acc = evaluate(model, test_dl, device)
        print(f"epoch {epoch}: train_loss={train_loss:.4f}  test_acc={test_acc * 100:.2f}%")

    print(f"total_time = {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
