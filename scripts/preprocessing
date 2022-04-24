dfs = {
    "train": pd.read_csv("/content/train.txt", delimiter=" "),
    "vali": pd.read_csv("/content/vali.txt", delimiter=" "),
    "test": pd.read_csv("/content/test.txt", delimiter=" "),
       }

for df in dfs.values():
  df.columns = np.arange(len(df.columns))
  df.drop(columns=df.columns[df.isna().all()].tolist(), inplace=True)

# Read in our datasets and remove columns that contain no data!

split = {}

split["X_train"] = dfs["train"].iloc[:, 1:]
split["X_vali"] = dfs["vali"].iloc[:, 1:]
split["X_test"] = dfs["test"].iloc[:, 1:]

y_train = dfs["train"].iloc[:, 0]
y_val = dfs["vali"].iloc[:, 0]
y_test = dfs["test"].iloc[:, 0]

g = split["X_train"].groupby(by=1) 
size = g.size()
train_group = size.to_list()

g = split["X_vali"].groupby(by=1)
size = g.size()
vali_group = size.to_list()

junk_columns = [41,42,43,44,45,66,67,68,69,70,
                91,92,93,94,95,16,17,18,19,20,
                71,72,73,74,75,76,77,78,79,80,
                81,82,83,84,85,86,87,88,89,90]

# Split our datasets according to our features and target variable
# Create subgroups group_train and group_vali containing samples per query ID to abide by LightGBM framework             
# In "Feature Selection and Model Comparison on Microsoft Learning-to-Rank Data Sets" (https://arxiv.org/pdf/1803.05127.pdf)
# Sen Lei and Xinzhi Han employ LASSO regression analysis to determine that variance and IDF-based features may be less 
# useful within our model. Therefore, we remove said features from our dataset

for name,df in split.items():
  df = df.astype(str)
  df = df.applymap(lambda x: x.split(":", 1)[-1]) #Clean up irrelevant info
  df = df.astype(float) # Convert data to float format for LGBMRanker
  df = df.drop(columns=1) # Remove query ID
  df.columns = [i for i in range(1,137)] # Rename columns for removal
  df.drop(columns=junk_columns)

  split[name] = df

print(split["X_train"],split["X_test"],split["X_vali"],y_train,y_test,y_val,vali_group,train_group)
