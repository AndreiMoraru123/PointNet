from tqdm import tqdm
from Transforms import *
from Dataset import *
from PointNet import *


def train_transforms():
    return transforms.Compose([PointSampler(1024), Normalize(), RandomRotation(), RandomNoise(), PointCloudToTensor()])


def test_transforms():
    return transforms.Compose([PointSampler(1024), Normalize(), PointCloudToTensor()])


def lossfun(outputs, labels, tr3x3, tr64x64, alpha=0.0001):
    criterion = nn.CrossEntropyLoss()
    bs = outputs.size()[0]
    id3x3 = torch.eye(3, requires_grad=True).repeat(bs, 1, 1)
    id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)
    if outputs.is_cuda:
        id3x3 = id3x3.cuda()
        id64x64 = id64x64.cuda()
    diff3x3 = id3x3 - torch.bmm(tr3x3, tr3x3.transpose(1, 2))
    diff64x64 = id64x64 - torch.bmm(tr64x64, tr64x64.transpose(1, 2))
    return criterion(outputs, labels) + alpha * (torch.norm(diff3x3) + torch.norm(diff64x64)) / float(bs)


train_dataset = PointCloudDataset(root=Path('ModelNet10'), mode='train', transform=train_transforms())
test_dataset = PointCloudDataset(root=Path('ModelNet10'), mode='test', transform=test_transforms())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('dataset_stats.txt', 'w') as f:
    f.write('Train dataset size: ' + str(len(train_dataset)) + '\n')
    f.write('Valid dataset size: ' + str(len(test_dataset)) + '\n')
    f.write('Number of classes: ' + str(len(train_dataset.classes)) + '\n')
    f.write('Sample pointcloud shape: ' + str(train_dataset[0]['pointcloud'].size()) + '\n')
    for k, v in train_dataset.classes.items():
        f.write(k + ': ' + str(v) + '\n')
    f.write('Device: ' + str(device) + '\n')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

model = PointNet(num_classes=len(train_dataset.classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
running_loss = 0.0

if __name__ == '__main__':

    with open('train_log.txt', 'w') as f:

        for epoch in range(25):
            model.train()
            for i, data in tqdm(enumerate(train_loader), total=len(train_loader)):

                inputs = data['pointcloud'].to(device)
                labels = data['category'].to(device)

                optimizer.zero_grad()
                outputs, m3x3, m64x64 = model(inputs.transpose(1, 2))

                loss = lossfun(outputs, labels, m3x3, m64x64)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 10 == 9:
                    f.write('Epoch: ' + str(epoch + 1) + ' Batch: ' + str(i) + ' Loss: ' + str(running_loss / 10) + '\n')
                    running_loss = 0.0

            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():

                for data in test_loader:
                    inputs = data['pointcloud'].to(device)
                    labels = data['category'].to(device)
                    outputs, _, _ = model(inputs.transpose(1, 2))
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            f.write('Epoch: ' + str(epoch + 1) + ' Accuracy: ' + str(100 * correct / total) + '\n')

            torch.save(model.state_dict(), 'model.pth')
