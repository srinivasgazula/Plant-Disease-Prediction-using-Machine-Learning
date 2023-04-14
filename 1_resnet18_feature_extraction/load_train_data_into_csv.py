#References:
#1. https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c
#2. https://stackoverflow.com/questions/61606416/runtimeerror-output-with-shape-512-doesnt-match-the-broadcast-shape-1-512


#Load the pretrained model
# model = models.resnet18(pretrained=True) #The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
#Use the model object to select the desired layer
layer = model._modules.get('avgpool')

#Set model to evaluation mode
model.eval()

#ResNet-18 expects images to be at least 224x224, as well as normalized with a specific mean and standard deviation
scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def extract_ResNet18_features(input_image_path):
  #Load the image with Pillow library
  img = Image.open(input_image_path)

  #Create a PyTorch Variable with the transformed image
  t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))

  #Create a vector of zeros that will hold our feature vector
    #The 'avgpool' layer has an output size of 512
  features_of_image = torch.zeros(512)

  #Define a function that will copy the output of a layer
  def copy_data(m, i, o):
    features_of_image.copy_(o.data.reshape(o.data.size(1)))
    # features_of_image.copy_(o.data)
  #Attach that function to our selected layer
  h = layer.register_forward_hook(copy_data)

  #Run the model on our transformed image
  model(t_img)

  # Detach our copy function from the layer
  h.remove()

  # Return the feature vector 
  return features_of_image.numpy()


path =  '/content/drive/MyDrive/EE769/project_dataset/train'

plants_list = []  #['Orange', 'Peach', 'Squash', 'Strawberry', 'Soyabean', 'Corn_maize', 'Apple', 'Grape', 'Potato', 'Pepper', 'Raspberry', 'Blueberry', 'Tomato', 'Cherry']
diseases_dict = {}
obj = os.scandir(path)
for entry in obj:
    if entry.is_dir():
        plants_list.append(entry.name)

print(plants_list)

features_array = []
target_plants_array = []
target_diseases_array = []

for x in plants_list:
  plants_path = path + str('/') + x
  temp_list = []
  obj = os.scandir(plants_path)
  for entry in obj:
    if entry.is_dir():
      temp_list.append(entry.name)
  diseases_dict[x] = temp_list
  for y in temp_list:
    disease_path = plants_path+ str('/') + y
    obj = os.scandir(disease_path)
    # count = 0
    for entry in obj:
      # if count<2:
      if entry.is_file():
        file_path = disease_path + str('/') + entry.name
        features_array.append(extract_ResNet18_features(file_path))
        target_plants_array.append(x)
        target_diseases_array.append(y)
      #     count += 1
      # else:
      #   break

rows = len(features_array)
cols = len(features_array[0])

print('-------------------------------------------------------------------\n')
print('Dimensions of features array list is ' + str(rows) + ' X ' + str(cols) + '\n')
print('-------------------------------------------------------------------\n')

#Converting feature data, target data into dataframes
features_list = []
for x in range(cols):
  features_list.append(str(x))
X = pd.DataFrame(features_array, columns = features_list)
y1_train = pd.DataFrame({'plant':target_plants_array})
y2_train = pd.DataFrame({'disease':target_diseases_array})

y1_classes = np.unique(y1_train) #plants
y2_classes = np.unique(y2_train) #diseases

print(y1_train.value_counts().reset_index())
print(y2_train.value_counts().reset_index())

# # # Multivariate feature imputation
# imp = IterativeImputer(max_iter=10, random_state=10)
# imp = imp.fit(X)
# X1_train = imp.transform(X)
# X1_train = pd.DataFrame(X1_train, columns = X.columns)

X1_train = X
X2_train = X1_train

df = pd.concat([X1_train, y1_train, y2_train], axis=1)
df.to_csv('/content/drive/MyDrive/EE769/data_tain.csv')

print('The obtained features data array is:')
display(pd.concat([X1_train, y1_train, y2_train], axis=1))
print('-------------------------------------------------------------------\n')
