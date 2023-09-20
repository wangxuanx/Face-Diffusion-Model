import torch
from datasets.data_loader import get_dataloaders

dataloaders = get_dataloaders(batch_size=1)

train_loader = dataloaders['train']
valid_loader = dataloaders['valid']
test_loader = dataloaders['test']

max = 0
min = 100000

for i, (motion, template, one_hot, file_name) in enumerate(train_loader):
    if torch.max(motion) > max:
        max = torch.max(motion)
        print('train max:', max)
    if torch.min(motion) < min:
        min = torch.min(motion)
        print('train min:', min)

for i, (motion, template, one_hot, file_name) in enumerate(valid_loader):
    if torch.max(motion) > max:
        max = torch.max(motion)
        print('valid max:', max)
    if torch.min(motion) < min:
        min = torch.min(motion)
        print('valid min:', min)

for i, (motion, template, one_hot, file_name) in enumerate(test_loader):
    if torch.max(motion) > max:
        max = torch.max(motion)
        print('test max:', max)
    if torch.min(motion) < min:
        min = torch.min(motion)
        print('test min:', min)

print('max:', max)
print('min:', min)