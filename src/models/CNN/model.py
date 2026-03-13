import torch
import torch.nn as nn
from pathlib import Path

name_mapping = {
    "tench": "n01440764",
    "English springer": "n02102040",
    "cassette player": "n02979186",
    "chain saw": "n03000684",
    "church": "n03028079",
    "French horn": "n03394916",
    "garbage truck": "n03417042",
    "gas pump": "n03425413",
    "golf ball": "n03445777",
    "parachute": "n03888257",
}

id_mapping = {v: k for k, v in name_mapping.items()}
idx_mapping = {idx: k for idx, k in enumerate(name_mapping.keys())}


class CNN(nn.Module):
    def __init__(self, num_classes, fe_dropout=0.1, cl_dropout=0.5):
        super(CNN, self).__init__()

        # Feature Extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1),  # 62x62x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 30x30x64
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # 30x30x192
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 14x14x192
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # 14x14x384
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # 14x14x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # 14x14x256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 6x6x256
            nn.Dropout(p=fe_dropout),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=cl_dropout),
            nn.Linear(256 * 6 * 6, 4096),  # FC1
            nn.ReLU(inplace=True),
            nn.Dropout(p=cl_dropout),
            nn.Linear(4096, 4096),  # FC2
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # FC3
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def load_model(path: str = "cnn_model.pth"):
    """
    Returns the model saved in CNN project folder
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_dir = Path(__file__).resolve().parent
    path = base_dir / "cnn_model.pth"
    data = torch.load(path, map_location=device)

    model = CNN(10)
    model.load_state_dict(data["model"])
    model.to(device)
    model.eval()

    return model, device


if __name__ == "__main__":
    from PIL import Image
    from pathlib import Path
    import random as rn
    import numpy as np
    from torchvision.transforms import v2
    import matplotlib.pyplot as plt

    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(128, 128)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    model, device = load_model(Path("./src/models/CNN/cnn_model.pth"))

    random_class = rn.choice(list(id_mapping.keys()))

    dir = Path("./src/datasets/imagenette2/train") / random_class
    print("Taking photo from: {} ({})".format(dir, id_mapping[random_class]))

    images = [file for file in dir.iterdir()]

    image = Image.open(rn.choice(images))
    input_image: torch.Tensor = transforms(image)
    input_image = input_image.to(device).unsqueeze(0)

    logits: torch.Tensor = model(input_image)
    logits = logits.cpu().detach()[0]
    predicts = torch.softmax(logits, dim=0).numpy()

    print(
        "Predicted Class: {}\n with total perception: {}".format(
            idx_mapping[np.argmax(predicts)],
            round(np.max(predicts), 2),
        )
    )

    print("Also less probable classes: ")

    for pred in np.argsort(predicts)[::-1]:
        print("  {:<20} {}".format(idx_mapping[pred], round(predicts[pred], 2)))

    plt.imshow(image)
    plt.title(
        f"pred: {idx_mapping[np.argmax(predicts)]}   orig: {id_mapping[random_class]}"
    )
    plt.show()
