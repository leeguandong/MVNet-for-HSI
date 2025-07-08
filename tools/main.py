import sys

sys.path.append("/dev_share/gdli7/common/HyperspectralCls_Factory/")

import os
import argparse
import collections
import datetime
import time
import numpy as np
import torch
from loguru import logger
from sklearn import preprocessing, metrics
from torch import optim

from src import (
    FeatherNet_network,
    DGCdenseNet,
    DydenseNet,
    LgcdenseNet,
    Densenet_Unet3D,
    CodenseNet,
    DsdenseNet,
    WTDenseNet,
    CGCdenseNet,
    KANet,
    STdenseNet,
    MambaDenseNet,
    # MambaDenseNet,
    load_dataset, sampling, generate_iter, aa_and_each_accuracy, record_output, generate_png)
from src.trainer import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_list = {
    'feathernet3d': FeatherNet_network,
    'dydenseNet': DydenseNet,
    'dgcnet': DGCdenseNet,
    'lgcdenseNet': LgcdenseNet,
    "densenet_unet3d": Densenet_Unet3D,
    "codensenet": CodenseNet,
    "dsdensenet": DsdenseNet,
    "wtdensenet": WTDenseNet,
    "cgcdenenet": CGCdenseNet,
    "kanet": KANet,
    "stnet": STdenseNet,
    # "mambanet": MambaDenseNet,
    "mambadensenet": MambaDenseNet
}


def parse_args():
    parser = argparse.ArgumentParser(description="data_process")
    parser.add_argument("--dataset", type=str, default="PaviaU",
                        choices=["Indian", "PaviaU", "Pavia", "KSC", "Botswana", "Indian"])
    parser.add_argument("--model", type=str, default="mambadensenet",
                        choices=['dydenseNet', 'dgcnet', "densenet_unet3d", "lgcnet", "codensenet", "dsdensenet",
                                 "wtdensenet", "cgcdenenet", "kanet", "stnet", "mambadensenet"])

    parser.add_argument("--iter", type=int, default=1)
    parser.add_argument("--path_length", type=int, default=5)
    parser.add_argument("--train_split", type=float, default=0.5)
    parser.add_argument("--num_epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=80)

    parser.add_argument("--saved", type=str, default='/dev_share/gdli7/common/HyperspectralCls_Factory/results/small')

    args = parser.parse_args()
    return args


def run(args, train_split, PATCH_LENGTH):
    seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341]

    ITER = args.iter
    lr = args.lr
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    model_name = args.model
    # import pdb;pdb.set_trace()
    saved = args.saved + f"/{model_name}/{str(train_split)}_{str(2 * PATCH_LENGTH + 1)}_{args.dataset}"

    if not os.path.exists(saved):
        os.makedirs(saved)

    # Set up logger
    log_file = os.path.join(saved, "training.log")
    logger.add(log_file)

    day_str = datetime.datetime.now().strftime('%m_%d_%H_%M')

    logger.info('-----Importing Dataset-----')
    data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, TRAIN_SPLIT = load_dataset(args.dataset, train_split)
    logger.info(f'The size of the HSI data is: {data_hsi.shape}')
    image_x, image_y, BAND = data_hsi.shape
    data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))
    gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )
    CLASSES_NUM = max(gt)
    logger.info(f'The class numbers of the HSI data is: {CLASSES_NUM}')

    img_rows = 2 * PATCH_LENGTH + 1
    img_cols = 2 * PATCH_LENGTH + 1
    img_channels = data_hsi.shape[2]
    INPUT_DIMENSION = data_hsi.shape[2]

    KAPPA = []
    OA = []
    AA = []
    TRAINING_TIME = []
    TESTING_TIME = []
    ELEMENT_ACC = np.zeros((ITER, CLASSES_NUM))

    data = preprocessing.scale(data)
    whole_data = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
    padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                             'constant', constant_values=0)

    for index_iter in range(ITER):
        net = net_list[args.model](BAND, CLASSES_NUM)
        logger.info(f"Model: {args.model}")
        net = torch.nn.DataParallel(net).cuda()

        loss = torch.nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.0001)
        # optimizer = optim.SGD(net.parameters(),lr=lr,weight_decay=0.0005)
        # optimizer = optim.RMSprop(net.parameters(),lr=lr)
        # optimizer = optim.Adagrad(net.parameters(),lr=lr,weight_decay=0.01)
        # optimizer = optim.Adadelta(net.parameters(),lr=lr)

        np.random.seed(seeds[index_iter])
        train_indices, test_indices = sampling(TRAIN_SPLIT, gt)
        _, total_indices = sampling(1, gt)

        TRAIN_SIZE = len(train_indices)
        logger.info(f'Train size: {TRAIN_SIZE}')
        TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
        logger.info(f'Test size: {TEST_SIZE}')
        VAL_SIZE = int(TOTAL_SIZE * 0.1)
        logger.info(f'Validation size: {VAL_SIZE}')

        logger.info('-----Selecting Small Pieces from the Original Cube Data-----')
        train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE,
                                                                     test_indices,
                                                                     TOTAL_SIZE, total_indices, VAL_SIZE,
                                                                     whole_data, PATCH_LENGTH, padded_data,
                                                                     INPUT_DIMENSION,
                                                                     batch_size, gt)

        tic1 = time.perf_counter()
        train(net, train_iter, valida_iter, loss, optimizer, device, saved, epochs=num_epochs)
        toc1 = time.perf_counter()

        pred_test = []
        tic2 = time.perf_counter()
        with torch.no_grad():
            for X, y in test_iter:
                X = X.to(device)
                net.eval()  # 评估模式, 这会关闭dropout
                y_hat = net(X)
                pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))
        toc2 = time.perf_counter()
        collections.Counter(pred_test)
        gt_test = gt[test_indices] - 1
        overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
        confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
        each_acc, average_acc = aa_and_each_accuracy(confusion_matrix)
        kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])

        torch.save(net.state_dict(), saved + "/" + str(round(overall_acc, 3)) + '.pth')
        KAPPA.append(kappa)
        OA.append(overall_acc)
        AA.append(average_acc)
        TRAINING_TIME.append(toc1 - tic1)
        TESTING_TIME.append(toc2 - tic2)
        ELEMENT_ACC[index_iter, :] = each_acc

    logger.info("--------" + net.module.name + " Training Finished-----------")

    record_output(OA, AA, KAPPA, ELEMENT_ACC, TRAINING_TIME, TESTING_TIME,
                  saved + '/' + net.module.name + day_str + '_' + args.dataset + 'split：' + str(
                      TRAIN_SPLIT) + 'lr：' + str(lr) + '.txt')

    generate_png(all_iter, net, gt_hsi, args.dataset, device, total_indices, saved)
    return np.mean(OA)


def main():
    args = parse_args()

    # 1. 先运行 train_split 的组合
    # train_split_list = [0.2, 0.3, 0.4, 0.5, 0.6]  # 替换为你需要的组合
    # overall_accuracies = {}  # Dictionary to store accuracies for each train_split0
    #
    # for train_split in train_split_list:
    #     logger.info(f'Running experiment with train_split = {train_split}')
    #     avg_oa = run(args, train_split, args.path_length)
    #     overall_accuracies[train_split] = avg_oa
    #     logger.info(f'Average Overall Accuracy for train_split {train_split}: {avg_oa:.4f}')

    # Find the best train_split based on highest average overall accuracy
    # best_train_split = max(overall_accuracies, key=overall_accuracies.get)
    # best_accuracy = overall_accuracies[best_train_split]
    # logger.info(f'Best train_split: {best_train_split} with OA: {best_accuracy:.4f}')
    best_train_split = 0.6
    # 2. Second stage: Run with different patch lengths using best_train_split
    # patch_length_list = [3, 4, 5, 6, 7, 8]
    patch_length_list = [7,8]
    for patch_length in patch_length_list:
        logger.info(f'Running experiment with patch_length = {patch_length} and best_train_split = {best_train_split}')
        run(args, best_train_split, patch_length)


if __name__ == "__main__":
    main()
