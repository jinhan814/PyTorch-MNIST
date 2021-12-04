from torchvision.datasets import MNIST
import json
import os
from tqdm import tqdm


download_path = '/opt/ml/MNIST/data'

train = MNIST(download_path, train=True, download=True)
test = MNIST(download_path, train=False, download=True)

print("train : ", len(train))
print("test  : ", len(test))

def save(path, n, bias, dataset):
    info = {}
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, 'image'), exist_ok=True)

    for i in tqdm(range(n)):
        img, label = dataset[bias + i]
        img_path   = os.path.join(path, 'image', str(i) + '.jpg')
        img.save(img_path)
        info[i] = { 'img_path': img_path, 'label': label }

    with open(os.path.join(path, 'info.json'), 'w') as f:
        json.dump(info, f, indent = 4)

# train, val, test
save(download_path + '/train', 50000, 0, train)
save(download_path + '/val', 10000, 50000, train)
save(download_path + '/test', 10000, 0, test)
