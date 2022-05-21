import pandas as pd
from ts_model import divide_train_test
from ts_model import ts_model
train_file_path = r"dataset/pollution.csv"
#data is univariate tite series
data = pd.read_csv(train_file_path, header=0, index_col=0)["pollution"].values
#get train_data and test_data
train_data, test_data = divide_train_test(data)
#initialise model parameters
ts = ts_model(cuda=False,n_epochs=25)
# fit and predict
preds, reals = ts.fit_transform(train_data, test_data)
# plot prediction result
ts.plot_predict_result(preds, reals)