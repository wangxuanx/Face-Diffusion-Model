import sys
sys.path.append('.')
import os
import torch
from zmq import device
from datasets.data_loader_mead import get_dataloaders
from tqdm import tqdm
from FLAME_PyTorch.FLAME import FLAME
from FLAME_PyTorch.config import get_config
from models.encoder_decoder import Trainer
from utiles.flame_utils import get_mesh, torch2mesh
from torch.utils.tensorboard import SummaryWriter

# 训练一个Motion Encoder和Motion Decoder来直接采样

def dataloader(batch_size=64, workers=10, read_audio=False):
    loader = get_dataloaders(batch_size=batch_size, workers=workers, read_audio=read_audio)
    train_loader = loader['train']
    val_loader = loader['valid']
    test_loader = loader['test']

    return train_loader, val_loader, test_loader

def main():
    save_path = './checkpoints/mead_encoder_decoder'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    writer = SummaryWriter(save_path)
    dev = 'cuda:1'
    load_model = False
    train_epochs = 400
    val_epoch = 11
    save_epoch = 100
    best_loss = 1000000  # 保存最好的loss

    flame_config = get_config()
    flame = FLAME(flame_config).to(dev)  # 加载FLAME模型
    for param in flame.parameters():
        param.requires_grad = False

    train_loader, val_loader, test_loader = dataloader(batch_size=1, workers=10, read_audio=False)

    model = Trainer(512, 5023 * 3).to(dev)  # where flame template is 5023 * 3, BIWI template is 23370 * 3 
    opt = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0)

    if load_model:
        load('./checkpoints/encoder_decoder', '400', model, opt)

    for epoch in range(train_epochs):
        epoch += 1
        model.train()
        print(f'Starting epoch {epoch}')
        train_loader = tqdm(train_loader, desc=f'Epoch[{epoch}/{train_epochs}]')
        batch_loss = []
        for i, (motion, template, one_hot, file_name) in enumerate(train_loader):
            opt.zero_grad()
            motion = motion.to(dev)
            template = template.to(dev)
            motion = torch2mesh(flame, motion[:, :, :50], motion[:, :, 50:])
            template = torch2mesh(flame, template[:, :, :50], template[:, :, 50:])

            loss = run_step(model, motion.flatten(-2), template.flatten(-2))
            batch_loss.append(loss.item())  # 添加batch loss

            loss.backward()
            opt.step()

            writer.add_scalar('Loss/training', loss, epoch * len(train_loader) + i)

            train_loader.set_postfix(loss=loss.item())  # 为tqdm添加loss项
            train_loader.update(1)
        
        batch_loss = sum(batch_loss) / len(batch_loss)  # 一个batch的平均loss
        print(f'Epoch {epoch} loss: {batch_loss}')
        writer.add_scalar('Loss/epoch', batch_loss, epoch)

        # 对模型进行评估
        # if epoch % val_epoch == 0:
        #     model.eval()
        #     val_loader = tqdm(val_loader, desc=f'Validing...')
        #     batch_loss = []
        #     for i, (motion, template, one_hot, file_name) in enumerate(train_loader):
        #         opt.zero_grad()
        #         motion = motion.to(dev)
        #         template = template.to(dev)

        #         loss = val_step(model, motion, template)
        #         batch_loss.append(loss.item())

        #         val_loader.set_postfix(loss=loss.item())  # 为tqdm添加loss项
        #         val_loader.update(1)

        #     batch_loss = sum(batch_loss) / len(batch_loss)  # 一个batch的平均loss
        #     writer.add_scalar('Loss/val', batch_loss, epoch)
        #     if batch_loss < best_loss:
        #         best_loss = batch_loss
        #         print(f'Best loss: {best_loss}')
        #         print(f'Saving best model...')
        #         save(save_path, 'best', model, opt)

        # 按照步数保存模型
        if epoch % save_epoch == 0:
            print(f'Saving model {epoch}...')
            save(save_path, epoch, model, opt)


def run_step(model, motion, template):
    recon = model(motion, template)  # 重建的motion
    loss = model.loss_fun(motion, recon)
    return loss

def val_step(model, motion, template):
    recon = model(motion, template)  # 重建的motion
    loss = model.loss_fun(motion, recon)
    return loss

def save(save_path, epoch, model, opt):
        data = {
            'model': model.state_dict(),
            'opt': opt.state_dict()
        }
        torch.save(data, str(save_path + f'/model-{epoch}.mpt'))

def load(save_path, epoch, model, opt):
    data = torch.load(str(save_path + f'/model-{epoch}.mpt'))
    model.load_state_dict(data['model'])
    opt.load_state_dict(data['opt'])



if __name__ == '__main__':
    main()