import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from scipy.stats import uniform


def onevsall_model(motions,df,model_directory):
    mask = ~(df['motion_type'].isin(motions))
    df.loc[mask,'motion_type'] = '0'

    dictio = {}
    cnt = 1
    for m in motions:
        dictio[m] = cnt
        cnt += 1

    df['motion_type'] = df['motion_type'].replace(dictio)
    df['motion_type'] = df['motion_type'].astype(int)

    train_set, test_set = train_test_split(df, test_size=0.2, stratify=df['motion_type'])

    y_train = train_set.loc[:,"motion_type"]
    X_train = train_set.drop('motion_type',axis=1)

    y_test = test_set.loc[:,"motion_type"]
    X_test = test_set.drop('motion_type',axis=1)


    scaler = StandardScaler()
    x_n = scaler.fit_transform(X_train)


    clf = VotingClassifier(
    estimators=[
        ('lr', OneVsRestClassifier(LogisticRegression(multi_class='ovr', solver='lbfgs'))),
        ('rf', OneVsRestClassifier(RandomForestClassifier())),
        ('svc', OneVsRestClassifier(SVC(kernel="rbf",probability=True)))
        ]
    )

    param_grid = {
        'lr__estimator__C': uniform(0.1, 10),  # Hyperparameter for Logistic Regression
        'svc__estimator__C': uniform(0.1, 10),  # Hyperparameter for SVM
        'svc__estimator__gamma': ['scale', 'auto'],  # Hyperparameter for SVM
        'rf__estimator__n_estimators': [50, 100, 150,200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],  # Hyperparameter for Random Forest
        'rf__estimator__bootstrap': [True, False],
        'rf__estimator__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'rf__estimator__max_features': ['log2', 'sqrt'],
        'rf__estimator__min_samples_leaf': [1, 2, 4],
        'rf__estimator__min_samples_split': [2, 5, 10],
        'voting': ['hard', 'soft']  # Voting type
    }

    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid,
                                    n_iter=10, cv=5, random_state=42, verbose=2, n_jobs=-1)

    random_search.fit(x_n, y_train)

    best_voting_clf = VotingClassifier(estimators=[('lr', OneVsRestClassifier(LogisticRegression(C=random_search.best_params_['lr__estimator__C'],multi_class='ovr', solver='lbfgs'))),
                                                    ('svm', OneVsRestClassifier(SVC(C=random_search.best_params_['svc__estimator__C'],
                                                                gamma=random_search.best_params_['svc__estimator__gamma'],kernel="rbf",probability=True))),
                                                    ('rf', OneVsRestClassifier(RandomForestClassifier(n_estimators=random_search.best_params_['rf__estimator__n_estimators'], bootstrap=random_search.best_params_['rf__estimator__bootstrap'], 
                                                                                    max_depth=random_search.best_params_['rf__estimator__max_depth'],max_features=random_search.best_params_['rf__estimator__max_features'],
                                                                                    min_samples_leaf=random_search.best_params_['rf__estimator__min_samples_leaf'], min_samples_split=random_search.best_params_['rf__estimator__min_samples_split'])))],
                                        voting=random_search.best_params_['voting'])


    print(random_search.best_params_)
    best_voting_clf.fit(x_n, y_train)

    x_t = scaler.transform(X_test)
    y_pred = best_voting_clf.predict(x_t)
    print(confusion_matrix(y_test, y_pred))

    path = Path(model_directory)
    path.mkdir(parents=True, exist_ok=True)
    model_f = 'model.sav'
    scaler_f = 'scaler.sav'
    joblib.dump(best_voting_clf, model_directory+"/"+model_f)
    joblib.dump(scaler, model_directory+"/"+scaler_f)

def multiclass_classifier(motions,df,model_directory):
    mask = ~(df['motion_type'].isin(motions))
    df.loc[mask,'motion_type'] = '0'

    dictio = {}
    cnt = 1
    for m in motions:
        dictio[m] = cnt
        cnt += 1

    df['motion_type'] = df['motion_type'].replace(dictio)
    df['motion_type'] = df['motion_type'].astype(int)

    train_set, test_set = train_test_split(df, test_size=0.2, stratify=df['motion_type'])

    y_train = train_set.loc[:,"motion_type"]
    X_train = train_set.drop('motion_type',axis=1)

    y_test = test_set.loc[:,"motion_type"]
    X_test = test_set.drop('motion_type',axis=1)


    scaler = StandardScaler()
    x_n = scaler.fit_transform(X_train)


    clf = RandomForestClassifier()

    param_grid = {
        'n_estimators': [50, 100, 150,200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],  # Hyperparameter for Random Forest
        'bootstrap': [True, False],
        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
        'max_features': ['log2', 'sqrt'],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
    }

    random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid,
                                    n_iter=100, cv=5, random_state=42, verbose=2, n_jobs=-1,scoring ='f1_macro')

    random_search.fit(x_n, y_train)

    
    best_model = RandomForestClassifier(n_estimators=random_search.best_params_['n_estimators'], bootstrap=random_search.best_params_['bootstrap'], 
                                                                                    max_depth=random_search.best_params_['max_depth'],max_features=random_search.best_params_['max_features'],
                                                                                    min_samples_leaf=random_search.best_params_['min_samples_leaf'], min_samples_split=random_search.best_params_['min_samples_split'])

    best_model.fit(x_n, y_train)

    x_t = scaler.transform(X_test)
    y_pred = best_model.predict(x_t)
    print(confusion_matrix(y_test, y_pred))

    path = Path(model_directory)
    path.mkdir(parents=True, exist_ok=True)
    model_f = 'model.sav'
    scaler_f = 'scaler.sav'
    joblib.dump(best_model, model_directory+"/"+model_f)
    joblib.dump(scaler, model_directory+"/"+scaler_f)
