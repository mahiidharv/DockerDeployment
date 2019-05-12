from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,classification_report, recall_score
#from sklearn.model_selection import GridSearchCV
from joblib import dump
from app import helpers as help
import category_encoders as ce
from app import core as c
#from core import c.X_train, c.y_train, c.X_val, c.y_val, c.numAttr,c.catAttr
import configparser
import datetime
path = help.path
config = configparser.ConfigParser()
config.read(path+'/config.ini')
print("############################################################################################################################################################")
print("Model Building")
print("############################################################################################################################################################")

numeric_transformer = Pipeline(memory='./', steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])


categorical_transformer = Pipeline(memory='./', steps=[
    ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
   ('onehot', ce.OneHotEncoder(use_cat_names=True, drop_invariant=True, handle_unknown='ignore'))
     ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, c.numAttr),
        ('cat', categorical_transformer, c.catAttr)])

# Model parameters after grid search 

C = int(config['Model']['C'])
kernel = config['Model']['kernel']
gamma = float(config['Model']['gamma'])

classifier = SVC(C=C,kernel=kernel,gamma=gamma,class_weight='balanced')

clf = Pipeline(memory='./', steps=[('preprocessor', preprocessor), ('classifier', classifier)])



print("############################################################################################################################################################")
print("Executing the classifier pipeline")
print("############################################################################################################################################################")

clf.fit(c.X_train, c.y_train)
y_pred_train = clf.predict(c.X_train)


print("############################################################################################################################################################")
print("Train Accuracy = ", accuracy_score(c.y_train, y_pred_train))
print("Recall in train = ", recall_score(c.y_train, y_pred_train, pos_label='yes'))
print("############################################################################################################################################################")

y_pred_val = clf.predict(c.X_val)

print("############################################################################################################################################################")
print("Validation Accuracy = ", accuracy_score(c.y_val, y_pred_val))
print("Recall on Validation = ", recall_score(c.y_val, y_pred_val, pos_label='yes'))
print("############################################################################################################################################################")

print("Classification Report")
print(classification_report(c.y_val, y_pred_val, digits=4))

print("############################################################################################################################################################")

currentDT = datetime.datetime.now()

dump_file = path+'pickle/model.pkl'+'_'+str(currentDT)
dump(clf, dump_file, compress=1)
print('\n Saved %s grid search pipeline to file: %s' % (clf, dump_file))
