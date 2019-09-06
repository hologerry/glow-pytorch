import os
import re
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSTOINS = [".png", ".jpg", ".jpeg", ".bmp"]
ATTR_ANNO = "attributes.txt"


def _is_image(fname):
    _, ext = os.path.splitext(fname)
    return ext.lower() in IMAGE_EXTENSTOINS


def _find_images_and_annotation(root_dir):
    images = {}
    attr = None
    assert os.path.exists(root_dir), "{} not exists".format(root_dir)
    for root, _, fnames in sorted(os.walk(root_dir)):
        for fname in sorted(fnames):
            if _is_image(fname):
                path = os.path.join(root, fname)
                images[os.path.splitext(fname)[0]] = path
            elif fname.lower() == ATTR_ANNO:
                attr = os.path.join(root, fname)

    assert attr is not None, "Failed to find `attributes.txt`"

    # begin to parse all image
    print("Begin to parse all image attrs")
    final = []
    with open(attr, "r") as fin:
        attrs = []
        for i_line, line in enumerate(fin):
            line = line.strip()
            if i_line == 0:
                attrs = line.split(" ")
            else:
                line = re.sub("[ ]+", " ", line)
                line = line.split(" ")
                fname = os.path.splitext(line[0])[0]
                char = int(fname.split('_')[1].split('.')[0])
                attr_vals = [(float(d)/100.0) for d in line[1:]]
                assert len(attr_vals) == len(attrs), "{} only has {} attrs < {}".format(
                    fname, len(attr_vals), len(attrs))
                final.append({
                    "path": images[fname],
                    "char": char,
                    "attr": attr_vals
                })
    print("Find {} images, with {} attrs".format(len(final), len(attrs)))
    return final, attrs


class ExploDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.Compose([
                                           transforms.CenterCrop(160),
                                           transforms.Resize(32),
                                           transforms.ToTensor()])):
        super().__init__()
        dicts, attrs = _find_images_and_annotation(root_dir)
        self.data = dicts
        self.attrs = attrs
        self.transform = transform

    def __getitem__(self, index):
        data = self.data[index]
        path = data["path"]
        char = data["char"]
        attr = data["attr"]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "x": image,
            "c": char,
            "y": np.asarray(attr, dtype=np.float32)
        }

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    import cv2
    Explo = ExploDataset("/home/chaiyujin/Downloads/Dataset/Explo")
    d = Explo[0]
    print(d["x"].size())
    img = d["x"].permute(1, 2, 0).contiguous().numpy()
    print(np.min(img), np.max(img))
    cv2.imshow("img", img)
    cv2.waitKey()
