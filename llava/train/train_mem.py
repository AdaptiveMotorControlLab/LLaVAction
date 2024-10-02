import sys
import os
sys.path[0] = os.path.dirname(os.path.dirname(sys.path[0]))
# sys.path.append(os.path.dirname(sys.path[0]))

from llava.train.train import train

if __name__ == "__main__":
    train()
