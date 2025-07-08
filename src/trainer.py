import time
import os
import numpy as np
import torch
from IPython import display
from matplotlib import pyplot as plt
from tqdm import tqdm
from loguru import logger


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize


def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            test_l_sum, test_num = 0, 0
            X = X.to(device)
            y = y.to(device)
            net.eval()  # 评估模式, 这会关闭dropout
            y_hat = net(X)
            l = loss(y_hat, y.long())
            acc_sum += (y_hat.argmax(dim=1) == y.long().to(device)).float().sum().cpu().item()
            test_l_sum += l.item()
            test_num += 1
            net.train()  # 改回训练模式
            n += y.shape[0]
    return [acc_sum / n, test_l_sum]  # / test_num]


def train(net, train_iter, valida_iter, loss, optimizer, device, saved, epochs=30, early_stopping=True, early_num=20):
    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    for epoch in range(epochs):
        train_acc_sum, n = 0.0, 0
        time_epoch = time.time()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
        # for X, y in tqdm(train_iter):
        for X, y in train_iter:
            batch_count, train_l_sum = 0, 0
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y.long()).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        lr_adjust.step(epoch)
        valida_acc, valida_loss = evaluate_accuracy(valida_iter, net, loss, device)
        loss_list.append(valida_loss)

        # 绘图部分
        train_loss_list.append(train_l_sum)  # / batch_count)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)

        logger.info(
            f'epoch: {epoch + 1}, train loss: {train_l_sum / batch_count}, train acc: {train_acc_sum / n}, '
            f'valida loss: {valida_loss}, valida acc: {valida_acc}, time {time.time() - time_epoch} sec')

        PATH = saved + "/" + "net.pth"

        if early_stopping and loss_list[-2] < loss_list[-1]:  # < 0.05) and (loss_list[-1] <= 0.05):
            if early_epoch == 0:  # and valida_acc > 0.9:
                torch.save(net.state_dict(), PATH)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                net.load_state_dict(torch.load(PATH))
                break
        else:
            early_epoch = 0

    set_figsize()
    plt.figure(figsize=(10, 10))
    train_accuracy = plt.subplot(221)
    train_accuracy.set_title('train_accuracy')
    plt.plot(np.linspace(1, epoch, len(train_acc_list)), train_acc_list, color='green')
    plt.xlabel('epoch')
    plt.ylabel('train_accuracy')
    plt.savefig(os.path.join(saved, "train_acc.png"))

    test_accuracy = plt.subplot(222)
    test_accuracy.set_title('valida_accuracy')
    plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue')
    plt.xlabel('epoch')
    plt.ylabel('test_accuracy')
    plt.savefig(os.path.join(saved, 'test_acc.png'))

    loss_sum = plt.subplot(223)
    loss_sum.set_title('train_loss')
    plt.plot(np.linspace(1, epoch, len(train_loss_list)), train_loss_list, color='red')
    plt.xlabel('epoch')
    plt.ylabel('train loss')
    plt.savefig(os.path.join(saved, 'train_loss.png'))

    test_loss = plt.subplot(224)
    test_loss.set_title('valida_loss')
    plt.plot(np.linspace(1, epoch, len(valida_loss_list)), valida_loss_list, color='gold')
    plt.xlabel('epoch')
    plt.ylabel('valida loss')
    plt.savefig(os.path.join(saved, 'test_loss.png'))

    # plt.show()
    logger.info(
        f'epoch: {epoch + 1}, loss: {train_l_sum / batch_count}, train acc: {train_acc_sum / n}, time: {time.time() - start} sec',
        )
