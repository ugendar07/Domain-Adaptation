

import torch

from Domain_Adversarial import *
from ResNet50 import Training_RealWorld,Test_RealWorld,Test_Clipart,Training_MNIST,Test_USPS


def get_hyperparameters() -> tuple[str , int, int, float]:
	DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
	BATCH_SIZE = 16
	NUM_EPOCHS = 15
	LEARNING_RATE = 1e-5

	print(DEVICE)

	return DEVICE,BATCH_SIZE,NUM_EPOCHS,LEARNING_RATE


def main() -> None:
    DEVICE,BATCH_SIZE,NUM_EPOCHS,LEARNING_RATE=get_hyperparameters()
    #Training_RealWorld(DEVICE ,BATCH_SIZE , NUM_EPOCHS , LEARNING_RATE)
    #Test_RealWorld(DEVICE,BATCH_SIZE,NUM_EPOCHS,LEARNING_RATE)
    #Test_Clipart(DEVICE , BATCH_SIZE )
    #Training_MNIST(DEVICE,BATCH_SIZE,NUM_EPOCHS,LEARNING_RATE)
    DANN_Training_OfficeHome(DEVICE ,BATCH_SIZE ,NUM_EPOCHS, LEARNING_RATE)
    #DANN_Training_MNIST(DEVICE ,BATCH_SIZE,NUM_EPOCHS,LEARNING_RATE)
    #DANN_Testing_USPS(DEVICE,BATCH_SIZE)
    #DANN_Testing_Clipart(DEVICE,BATCH_SIZE)







if __name__ == '__main__':
    main()

