import model
import reader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
import sys
import os
import copy
import yaml
import importlib

def gazeto3d(gaze):
  gaze_x = (-torch.cos(gaze[:, 1]) * torch.sin(gaze[:, 0])).unsqueeze(1)
  gaze_y = (-torch.sin(gaze[:, 1])).unsqueeze(1)
  gaze_z = (-torch.cos(gaze[:, 1]) * torch.cos(gaze[:, 0])).unsqueeze(1)
  gaze3d = torch.cat([gaze_x, gaze_y, gaze_z], 1)
  return gaze3d

def loss_op(gaze, label, device):
  totals = torch.sum(gaze*label, dim=1, keepdim=True)
  length1 = torch.sqrt(torch.sum(gaze*gaze, dim=1, keepdim=True))
  length2 = torch.sqrt(torch.sum(label*label, dim=1, keepdim=True))
  res = totals/(length1 * length2)
  angular = torch.mean(torch.acos(torch.min(res , torch.ones_like(res).to(device)*0.9999999)))*180/math.pi
  return angular

if __name__ == "__main__":
  config = yaml.load(open(sys.argv[1]), Loader=yaml.FullLoader)
  dataloader =  importlib.import_module("reader." + config['reader'])
  config = config["train"]

  imagepath = config["data"]["image"]
  labelpath = config["data"]["label"]
  modelname = config["save"]["model_name"]

  folder = os.listdir(labelpath)
  folder.sort()

  # i represents the i-th folder used as the test set.
  i = int(sys.argv[2])

  if i in list(range(15)):
    trains = copy.deepcopy(folder)
    tests = trains.pop(i)
    print(f"Train Set:{trains}")
    print(f"Test Set:{tests}")

    trainlabelpath = [os.path.join(labelpath, j) for j in trains] 

    savepath = os.path.join(config["save"]["save_path"], f"checkpoint/{tests}")
    if not os.path.exists(savepath):
      os.makedirs(savepath)
  
    device = torch.device("cuda:1") 
    
    print("Read data")
    dataset = dataloader.txtload(trainlabelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=8, header=True)

    print("Model building")
    net = model.model()
    net.train()
    net.to(device)

    print("optimizer building")
    loss_op = nn.MSELoss().cuda()
    #loss_op = getattr(nn, lossfunc)().cuda()
    
    base_lr = config["params"]["lr"]

    decaysteps = config["params"]["decay_step"]
    decayratio = config["params"]["decay"]

    optimizer = optim.Adam(net.parameters(),lr=base_lr, betas=(0.9,0.95))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decaysteps, gamma=decayratio)

    print("Training")
    length = len(dataset)
    total = length * config["params"]["epoch"]
    cur = 0
    timebegin = time.time()
    with open(os.path.join(savepath, "train_log"), 'w') as outfile:
      for epoch in range(1, config["params"]["epoch"]+1):
        for i, (data, label) in enumerate(dataset):

          # Acquire data
          data["eye"] = data["eye"].to(device)
          data['head_pose'] = data['head_pose'].to(device)
          label = label.to(device)
   
          # forward
          gaze = net(data)

          # loss calculation
          # loss = loss_op(gazeto3d(gaze), gazeto3d(label), device)
          
          loss = loss_op(gaze, label)
          optimizer.zero_grad()

          # backward
          loss.backward()
          optimizer.step()
          scheduler.step()
          cur += 1

          # print logs
          if i % 20 == 0:
            timeend = time.time()
            resttime = (timeend - timebegin)/cur * (total-cur)/3600
            log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss} lr:{base_lr}, rest time:{resttime:.2f}h"
            print(log)
            outfile.write(log + "\n")
            sys.stdout.flush()   
            outfile.flush()

        if epoch % config["save"]["step"] == 0:
          torch.save(net.state_dict(), os.path.join(savepath, f"Iter_{epoch}_{modelname}.pt"))

