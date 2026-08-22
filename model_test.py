import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import AlexNet


def test_data_process():
    test_data = FashionMNIST(
        root="./data",
        train=False,
        transform=transforms.Compose([
            transforms.Resize(size=227),
            transforms.ToTensor()
        ]),
        download=True
    )

    test_dataloader = Data.DataLoader(
        dataset=test_data,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )
    return test_dataloader


def test_model_process(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    test_corrects = 0
    test_num = 0
    results = []  # 存储前15条推理结果

    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)

            test_corrects += torch.sum(pre_lab == b_y)
            test_num += b_x.size(0)

            # 收集前15个结果（预测+实际）
            if len(results) < 15:
                for i in range(b_x.size(0)):
                    if len(results) >= 15:
                        break
                    pred_label = classes[pre_lab[i].item()]
                    true_label = classes[b_y[i].item()]
                    results.append((pred_label, true_label))

    for idx, (pred, true) in enumerate(results, start=1):
        status = "✅" if pred == true else "❌"
        print(f"{idx:>2d}. Predicted: {pred:<12} | Actual: {true:<12} {status}")

    test_acc = test_corrects.double().item() / test_num
    print(f"\nTest accuracy: {test_acc:.4f}")

    return test_acc


if __name__ == '__main__':
    model = AlexNet()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    test_dataloader = test_data_process()

    test_model_process(model, test_dataloader)
