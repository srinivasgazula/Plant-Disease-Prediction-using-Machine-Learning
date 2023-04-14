def initialize_train_valid_data():
  data_train = pd.read_csv('/content/drive/MyDrive/EE769/data_tain.csv')

  # data_train = pd.read_csv('/content/drive/MyDrive/EE769/data_test.csv')

  features_list = list( data_train.loc[:][:1] )[1:-2]
  target_list  = list( data_train.loc[:][:1] )[-2:]

  #Extracting feature data from csv
  X1_train  = data_train.loc[:,features_list]
  X2_train = X1_train

  #Extracting plants target data from csv
  y1_train = data_train.loc[:,target_list[0]]
  #Extracting diseases target data from csv
  y2_train = data_train.loc[:,target_list[1]]

  y1_classes = np.unique(y1_train) #plants
  y2_classes = np.unique(y2_train) #diseases


  # ------------------------------------------
  data_valid = pd.read_csv('/content/drive/MyDrive/EE769/data_valid.csv')

  features_list = list( data_valid.loc[:][:1] )[1:-2]
  target_list  = list( data_valid.loc[:][:1] )[-2:]

  #Extracting feature data from csv
  X1_valid  = data_valid.loc[:,features_list]
  X2_valid = X1_valid

  #Extracting plants target data from csv
  y1_valid = data_valid.loc[:,target_list[0]]
  #Extracting diseases target data from csv
  y2_valid = data_valid.loc[:,target_list[1]]

  y1_valid_classes = np.unique(y1_valid) #plants
  y2_valid_classes = np.unique(y2_valid) #diseases
  return X1_train,y1_train,X2_train,y2_train,X1_valid,y1_valid,X2_valid,y2_valid,y1_classes,y2_classes,y1_valid_classes,y2_valid_classes
###--END OF FUNCTION--############
  
X1_train,y1_train,X2_train,y2_train,X1_valid,y1_valid,X2_valid,y2_valid,y1_classes,y2_classes,y1_valid_classes,y2_valid_classes = initialize_train_valid_data()

print(y1_train.value_counts().reset_index())
print(y2_train.value_counts().reset_index())
print('-------------------------------------------------------------------\n')
display(pd.concat([X1_train, y1_train, y2_train], axis=1))
print('-------------------------------------------------------------------\n')
print(y1_valid.value_counts().reset_index())
print(y2_valid.value_counts().reset_index())
print('-------------------------------------------------------------------\n')
display(pd.concat([X1_valid, y1_valid, y2_valid], axis=1))
