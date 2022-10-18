from DataLoader import *
from PointNet import *
from train import test_dataset
import matplotlib.pyplot as plt

model = PointNet(num_classes=len(test_dataset.classes)).to('cpu')
model.load_state_dict(torch.load('model.pth'))


def plot_classification(idx):
    sample = test_dataset[idx]
    pointcloud = sample['pointcloud']
    category = sample['category']
    pointcloud = pointcloud.unsqueeze(0)
    with torch.no_grad():
        model.eval()
        output, _, _ = model(pointcloud.transpose(1, 2))
        pred = output.max(1, keepdim=True)[1]

    fig = plt.figure()
    fig.suptitle('Predicted: ' + test_dataset.folders[pred.item()] + '\nActual: ' + test_dataset.folders[category])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pointcloud[0, :, 0], pointcloud[0, :, 1], pointcloud[0, :, 2], c=pointcloud[0, :, 2])
    plt.show()


index = np.random.randint(0, len(test_dataset))
plot_classification(index)