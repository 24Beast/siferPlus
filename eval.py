# Importing Libraries
import torch
from utils.data import CelebATarget
from torch.utils.data import DataLoader

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = "./models/target_5_forget_100_b_256/epoch_5/"
B_SIZE = 192

# Loading Models
pre_model = torch.load(MODEL_DIR+"pre.pth").to(DEVICE)
aux_model = torch.load(MODEL_DIR+"aux.pth").to(DEVICE)
main_model = torch.load(MODEL_DIR+"main.pth").to(DEVICE)

# Loading Data
test_data = CelebATarget("./data/", split="test")
test_dataloader = DataLoader(test_data, batch_size=B_SIZE, shuffle=False)

# Evaluation loop
main_male_acc = 0
aux_male_acc = 0
main_female_acc = 0
aux_female_acc = 0
total_males = 0
total_females = 0
print(f"Working for {MODEL_DIR=}")
for i, data in enumerate(test_dataloader):
    print(f"\rWorking on batch: {i}",end="")
    inputs, labels, targets = data
    inputs = inputs.to(DEVICE)
    labels = labels.to(DEVICE)
    pre_out = pre_model(inputs)
    main_out = main_model(pre_out)
    aux_out, aux_target = aux_model(pre_out)
    main_blondes = torch.argmax(main_out,axis=1)==1
    aux_blondes = torch.argmax(aux_out,axis=1)==1
    label_blondes = labels[:,1]
    main_correct = (main_blondes == label_blondes)
    aux_correct = (aux_blondes == label_blondes)
    num_males = torch.sum(targets)
    main_male_acc += torch.sum(main_correct[targets==1])
    aux_male_acc += torch.sum(aux_correct[targets==1])
    main_female_acc += torch.sum(main_correct[targets==0])
    aux_female_acc += torch.sum(aux_correct[targets==0])
    total_males += num_males
    total_females += targets.shape[0] - num_males
    
print(f"{main_male_acc/total_males=}")
print(f"{main_female_acc/total_females=}")

print(f"{aux_male_acc/total_males=}")
print(f"{aux_female_acc/total_females=}")
    
