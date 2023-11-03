from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from pandas import concat, DataFrame, read_csv
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

label_encoder = None 

def get_np_array(file_name):
    global label_encoder
    data = read_csv(file_name)
    
    need_label_encoding = ['team','host','opp','month', 'day_match']
    if(label_encoder is None):
        label_encoder = OneHotEncoder(sparse_output = False)
        label_encoder.fit(data[need_label_encoding])
    data_1 = DataFrame(label_encoder.transform(data[need_label_encoding]), columns = label_encoder.get_feature_names_out())
    
    #merge the two dataframes
    dont_need_label_encoding =  ["year","toss","bat_first","format" ,"fow","score" ,"rpo" ,"result"]
    data_2 = data[dont_need_label_encoding]
    final_data = concat([data_1, data_2], axis=1)
    
    X = final_data.iloc[:,:-1]
    y = final_data.iloc[:,-1:]
    return X.to_numpy(), y.to_numpy()


X_train,Y_train = get_np_array('train.csv')
X_test, Y_test = get_np_array("test.csv")
X_val, Y_val = get_np_array("val.csv")
Y_train = Y_train.flatten()
Y_test = Y_test.flatten()
Y_val = Y_val.flatten()

param_grid = {
  'n_estimators': [50, 150, 250, 350],
  'max_features': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
  'min_samples_split': [2, 4, 6, 8, 10]
}

model = RandomForestClassifier(oob_score=True, random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, Y_train)
print("Best Hyperparameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_
print("Training Accuracy:", best_model.score(X_train, Y_train))
print("Test Accuracy:", best_model.score(X_test, Y_test))
print("Validation Accuracy:", best_model.score(X_val, Y_val))
oob_accuracy = best_model.oob_score_
print("Out-of-Bag Accuracy:", oob_accuracy)