#!/usr/bin/env python
import argparse
import os
import matplotlib.pyplot as plt


def figure(file):
    """
    extracts information from file, generates a graph and saves it as file.png
    """
    embedding_stats= list()
    hit_at_1 = ''
    batch_size = ''

    with open(file, 'r') as f:
        for line in f:
            if "ComplEx - Epoch" in line:
                embedding_stats.append(line.strip())
            elif "Hit@1 :" in line:
                hit_at_1 = line.split('INFO:')[-1]
            elif "batch_size : " in line:
                batch_size = line.split(':')[-1].strip()

    train_loss = list()
    valid_loss = list()

    for element in embedding_stats:
        stats = element.split('|')[3]
        train_loss.append(float(stats.split(' ')[3].strip(',')))
        valid_loss.append(float(stats.split(' ')[6].strip()))

    #train line
    xt = list(range(1, len(train_loss)+1, 1))
    yt = train_loss
    plt.plot(xt, yt, label = "training loss")
    #validation line
    xv = xt
    yv = valid_loss
    plt.plot(xv, yv, label = "validation loss")
    #plot Hit @1 value
    plt.text(0.5, 0.5, hit_at_1)
    #plot batch size
    plt.text(0.5, 10, 'Batch_size = ' + batch_size)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(file)
    plt.legend()
    plt.savefig(file + '.png')

def main():
    """
    Generates a graph of the training and evaluation loss by epoch if grapoh file does not exist.
    graph also includes batch size and Hit@1 score of the final model
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('file', help='log file to extract informations from')

    args = parser.parse_args()

    logfile = args.file
    if not os.path.isfile(logfile + '.png'):
        figure(logfile)

if __name__ == "__main__":
    main()