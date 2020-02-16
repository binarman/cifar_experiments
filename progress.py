#!/usr/bin/python3

import matplotlib.pyplot as plt
import re
import sys

def gather_accuracy(in_file):
  epoch_acc_re = re.compile(".*val_acc: (.*)")
  acc_list = []
  for line in open(in_file):
    match = epoch_acc_re.match(line)
    if match:
      acc_list += [float(match.group(1))]
  return acc_list


def main(paths):
  for path in paths:
    print(path)
    points = gather_accuracy(path)
    print(points)
    plt.plot(points)
  plt.legend(paths)
  plt.grid(True)
  plt.show()


if __name__ == "__main__":
  main(sys.argv[1:])

