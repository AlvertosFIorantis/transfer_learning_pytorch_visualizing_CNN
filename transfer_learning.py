#%%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.autograd import Variable
from torchvision import datasets, models, transforms
import os
import numpy as np
import torchvision
#%%
#Data augmentation and normalization for training
# hust normalization for validation
# transforms.RandomResizedCrop(224) --> a crop of random size (default: of 0.08 to 1.0) of the original size and a
# random aspect ration (defalt: of 3/4 to 4/3) of the original aspect ratio is made.
#this crop is finally resized to given size (224 in this case) 224 is the output size
# transforms.CentreCrop(224) --> crops the image at the centre , 224 is the Desired output size of the crop

data_transforms = {
    "train":torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    "val": torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}
#%%
data_dir = "./"

# %%
#create a dictionary that contains the information of the images in both the training and validation set
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ["train","val"]}
#create a dictionary that contains the data loader
dataloaders = {x:torch.utils.data.DataLoader(image_datasets[x],
                                             batch_size=4,
                                             shuffle=True)for x in ["train","val"]}
#create a dictionary that contains the size of each datset (training and validation)
datasets_sizes= {x: len(image_datasets[x]) for x in ["train","val"]}
#Get the clas names
class_names= image_datasets["train"].classes
#print out the results
print("Class Names: {}".format(class_names))
print("Thera are {} batches in the training set".format(len(dataloaders["train"])))
print("there are {} batches in the test set".format(len(dataloaders["val"])))
print("There are {} training images".format(datasets_sizes["train"]))
print("there are {} testing images".format(datasets_sizes["val"]))
# we make the labels based on hte names of the folderes

# %%
# load the ResNet
model_conv = torchvision.models.resnet50(pretrained=True)

# %%
model_conv.parameters #gia na do pos einai ftiaxmeno to modelo

# %%
#Freeze all layers in the network
for param in model_conv.parameters():
  param.requires_grad = False

# %%
#Get the number of inputs of the last layer (or number of neurons in the layer preceeding the last layer)
num_ftrs = model_conv.fc.in_features
#Reconstruc the last layer (output layer) to the have only two classes
model_conv.fc = nn.Linear(num_ftrs,2)#vazo 2 giati ego exo dio classes eno to arxiko modelo eixei 1000!!!

# %%
if torch.cuda.is_available():
  model_conv = model_conv.cuda()

# %%
#Understanding what is happening
iteration = 0
correct = 0
for inputs,labels in dataloaders["train"]:
  if iteration ==1:
    break
  inputs = Variable(inputs)
  labels = Variable(labels)
  if torch.cuda.is_available():
    inputs = inputs.cuda()
    labels = labels.cuda()
  print("For one iteration this is what happens:")
  print("inputs shape:",inputs.shape)
  print("labels shape:",labels.shape)
  print("labels are: {}".format(labels))
  output = model_conv(inputs)
  print("output tensor:",output)
  print("Outputs shape",output.shape)
  _,predicted = torch.max(output,1)
  print("Predicted:",predicted)
  print("Predicted shape",predicted.shape)
  correct += (predicted == labels.sum())
  print("Correct Predictions:",correct)
  
  iteration +=1
#random guess 62.2 % on traiing
# %%
output = model_conv(inputs)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv.fc.parameters(),lr=0.001,momentum=0.9)
#Try experimenting with optim.Adam(model_conv.fc.parateres(),lr=0.001)
#Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=7,gamma=0.1)

# %%
#this is to demonstrate what happens in the background of scheduler.step()
#no need to run this cell unless you want to create your own scheduler
def lr_scheduler(optimizer,epoch,init_lr=0.001,lr_decay_epoch=7):
  #Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
  lr=init_lr*(0.1**(epoch//lr_decay_epoch))
  
  if epoch % lr_decay_epoch ==0:
    print("LR is set to {})".format(lr))
  
  for param_group in optimizer.param_groups:
    param_group["Lr"]=lr
  
  return optimizer

# %%
num_epochs = 25
for epoch in range(num_epochs):
  exp_lr_scheduler.step()
  #reset the correct 0 after passing through all the dataset
  correct = 0
  for images,labels in dataloaders["train"]:
    images=Variable(images)
    labels=Variable(labels)
    if torch.cuda.is_available():
      images=images.cuda()
      labels=labels.cuda()
    
    optimizer.zero_grad()
    outputs=model_conv(images)
    loss=criterion(outputs,labels)
    loss.backward()
    optimizer.step()
    _,predicted = torch.max(outputs,1)
    correct+=(predicted==labels).sum()
  
  train_acc=100*correct/datasets_sizes["train"]
  print("Epoch [{}/{}],Loss: {:.4f},Train Accuracy: {}%".format(
          epoch+1,num_epochs,loss.item(),train_acc))

# %%
#Test the model 
model_conv.eval()
with torch.no_grad():
  correct =0
  total=0
  for (images,labels) in dataloaders["val"]:
    images=Variable(images)
    labels=Variable(labels)
    if torch.cuda.is_available():
      images=images.cuda()
      labels=labels.cuda()
    
    outputs=model_conv(images)
    _,predicted = torch.max(outputs.data,1)
    total+=labels.size(0)
    correct +=(predicted==labels).sum().item()
  
  print("Test accuracy: {:.3f}%".format(100*correct/total))
#%%
#Visualize some predictions
import matplotlib.pyplot as plt
fig = plt.figure()
shown_batch=0
index=0
with torch.no_grad():
  for(images,labels) in dataloaders["val"]:
    if shown_batch ==1:
      break
    shown_batch +=1
    images = Variable(images)
    labels = Variable(labels)
    if torch.cuda.is_available():
      images = images.cuda()
      labels= labels.cuda()
    
    outputs= model_conv(images) #the output is of shape(4,2)
    _,preds = torch.max(outputs,1) #the pred is of shape(4) --> [0,0,0,1]
    
    for i in range(4):
      index+=1
      ax=plt.subplot(2,2,index)
      ax.axis("off")
      ax.set_title("Predicted label: {}".format(class_names[preds[i]]))
      input_img = images.cpu().data[i] #get teh tensor of the image and put it to cpu
      inp =input_img.numpy().transpose((1,2,0))
      mean=np.array([0.485,0.456,0.406])
      std = np.array([0.229,0.224,0.225])
      inp = std*inp+mean
      inp = np.clip(inp,0,1)
      plt.imshow(inp)

# %%
'''test'''
data_dir = "./"
data_transforms_test =  {
    "test":torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
  
}
#%%
image_datasets_test = {'test':datasets.ImageFolder(os.path.join(data_dir,'test'),data_transforms_test['test'])}
#create a dictionary that contains the data loader
#%%
dataloaders = {x:torch.utils.data.DataLoader(image_datasets_test[x],
                                             batch_size=4,
                                             shuffle=True)for x in ['test']}
#create a dictionary that contains the size of each datset (training and validation)
datasets_sizes= {x: len(image_datasets_test[x]) for x in ['test']}
#%%
#Get the clas names
fig = plt.figure()
shown_batch=0
index=0
with torch.no_grad():
  for(images,labels) in dataloaders["test"]:
    if shown_batch ==1:
      break
    shown_batch +=1
    images = Variable(images)
    labels = Variable(labels)
    if torch.cuda.is_available():
      images = images.cuda()
      labels= labels.cuda()
    
    outputs= model_conv(images) #the output is of shape(4,2)
    _,preds = torch.max(outputs,1) #the pred is of shape(4) --> [0,0,0,1]
    
    for i in range(4):
      index+=1
      ax=plt.subplot(2,2,index)
      ax.axis("off")
      ax.set_title("Predicted label: {}".format(class_names[preds[i]]))
      input_img = images.cpu().data[i] #get teh tensor of the image and put it to cpu
      inp =input_img.numpy().transpose((1,2,0))
      mean=np.array([0.485,0.456,0.406])
      std = np.array([0.229,0.224,0.225])
      inp = std*inp+mean
      inp = np.clip(inp,0,1)


 

#%%
# %%
'''New visualization layers part '''
module_list = list(model_conv.modules())
transform = transforms.Compose([ torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                               ])
image=Image.open("./relative path to image.JPG").convert('RGB')

#mage=Image.open("./test/test/er_photo_142912.jpg").convert('RGB')
plt.imshow(image)

#%%
image = transform(image)
#%%
image = image.unsqueeze(0)

# %%
#Wrap it up in a varaible
image = Variable(image)

#%%
#model_conv.layer[-1]
#%%
output = model_conv(image)

# %%
outputs = []
names = []
#%%
test=0
conv2dLayers = [num for num in module_list if type(num) == torch.nn.modules.conv.Conv2d]
children=list(model_conv.named_children())
#defaulght module_list
for layer in module_list[1:]:
  #print(layer)
  # image = layer(image)
  # outputs.append(image)
  # names.append(str(layer))
  try:
    image = layer(image)
    outputs.append(image)
    names.append(str(layer))
    test+=1
  except:
    print(test)

# %%
processed = []
for feature_map in outputs:
  feature_map = feature_map.squeeze(0)
  #convert the 3d tensro to 2d sum the same element of every channel
  gray_scale = torch.sum(feature_map,0)
  gray_scale = gray_scale / feature_map.shape[0] #nirmalize the gray scale
  processed.append(gray_scale.data.cpu().numpy())

# %%
demo_list=processed[0::5]#kai pernodas oli ti lsat boro na dikso oles ta layers

fig = plt.figure(figsize = (30,50))#kai afto

for i in range(len(demo_list)):
  a = fig.add_subplot(8,4,i+1)#alazodas afto 
  imgplot = plt.imshow(demo_list[i])
  plt.axis("off")
  a.set_title(names[0].split("(")[0],fontsize=30)

