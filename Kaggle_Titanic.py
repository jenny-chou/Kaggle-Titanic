# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 15:19:17 2020

@author: jenny
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, \
    train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score


class Log:
    # Log helper class.

    LOG_LEVEL = dict((value, index) for (index, value) in 
                     enumerate(["INFO", "ERROR", "FATEL"]))
    LEVEL = LOG_LEVEL["INFO"]
    
    @classmethod
    def set_log_level(cls, level="INFO"):
        """
        Set permissible log level.

        Args:
            level (str, optional): level to be set. Defaults to "INFO".

        Returns:
            None.
            
        """
        cls.LEVEL = cls.LOG_LEVEL[level]
    
    @classmethod
    def print_msg(cls, level, msg):
        """
        Check log level and print permissible logs.

        Args:
            level (str): level of the message.
            msg (str): message to be print.

        Returns:
            None.
            
        """
        if cls.LOG_LEVEL[level] >= cls.LEVEL:
            print(msg)        


class Data:
    # data container
    
    def __init__(self, train, test):
        self.train = train
        self.test = test
        
    def _encode_sex(self):
        """
        Transform Sex feature to numerical feature.

        Returns:
            None.

        """
        self.train.Sex = (self.train.Sex == 'female').astype(int)
        self.test.Sex = (self.test.Sex == 'female').astype(int)
        
    def _fill_null_age(self, df, age_dict, age_median):
        """
        Age feature imputation.

        Args:
            df (DataFrame): dataframe to be worked on.
            age_dict (dict): dictionary of age for each combination of Pclass 
                and Sex.
            age_median (float): 

        Returns:
            None.

        """
        values = []
        for ppl in df[df.Age.isnull()][['Pclass', 'Sex']].values:
            if tuple(ppl) in age_dict:
                values.append(age_dict[tuple(ppl)])
            else:
                values.append(age_median)
                
        df.loc[df.Age.isnull(), 'Age'] = values
        
    def _fill_age(self):
        """
        Age imputation.

        Returns:
            None.

        """
        class_sex_dict = \
            dict(self.train.groupby(['Pclass', 'Sex']).Age.median())
        age_median = self.train.Age.median()        
        
        self._fill_null_age(self.train, class_sex_dict, age_median)
        self._fill_null_age(self.test, class_sex_dict, age_median)
        
    def _replace_zero(self, df, feature):
        """
        Replace zero in feature with null.

        Args:
            df (DataFrame): dataframe to be worked on.
            feature (str): feature name.

        Returns:
            None.

        """
        df.loc[df[feature] == 0, feature] = np.NaN
    
    def _fill_null_fare(self, df, fare_dict):
        """
        Fare imputation.

        Args:
            df (DataFrame): dataframe to be worked on.
            fare_dict (dict): dictionary of fare for each Pclass.

        Returns:
            None.

        """
        df.loc[df.Fare.isnull(), "Fare"] = \
            df[df.Fare.isnull()].Pclass.map(fare_dict)
        
    def _fill_fare(self):
        """
        Fare imputation.

        Returns:
            None.

        """
        self._replace_zero(self.train, "Fare")
        self._replace_zero(self.test, "Fare")
        
        fare_dict = dict(self.train.groupby(['Pclass']).Fare.median())
        
        self._fill_null_fare(self.train, fare_dict)
        self._fill_null_fare(self.test, fare_dict)        
    
    def _fill_embark(self):
        """
        Embarked imputation.

        Returns:
            None.

        """
        self.train.Embarked.fillna("S", inplace=True)
        self.test.Embarked.fillna("S", inplace=True)

    def _encode_embark(self):
        """
        Transform Embarked to numerical values.

        Returns:
            None.

        """
        encoder = LabelEncoder().fit(self.train.Embarked)
        self.train.Embarked = encoder.transform(self.train.Embarked)
        self.test.Embarked = encoder.transform(self.test.Embarked)
        
        
    def preprocessing(self):
        """
        Data cleaning and data imputation.

        Returns:
            None.

        """
        self._encode_sex()
        self._fill_age()
        self._fill_fare()
        self._fill_embark()
        self._encode_embark()
        

class DataProcessing(Data):
    # Feature Engineering and feature selection. Inherit from Data.
    
    def __init__(self, train, test):
        super().__init__(train, test)
        self.feat_sel_matrix = pd.DataFrame()

    def _extract_ticket_num_df(self, df): 
        """
        Extract the number from Ticket feature in given dataframe.

        Args:
            df (DataFrame): dataframe to be worked on.

        Returns:
            None.

        """
        numbers = []
        for ticket in df.Ticket.values:
            if ticket == "LINE":
                numbers.append(0)
            for subs in ticket.split():
                if subs.isdecimal():
                    numbers.append(subs)
                    break
        df['TicketNum'] = np.array(numbers, dtype=int)
        
    def _extract_ticket_num(self):
        """
        Extract the number from Ticket feature.

        Returns:
            None.

        """
        self._extract_ticket_num_df(self.train)
        self._extract_ticket_num_df(self.test)
    
    def _add_family_size(self):
        """
        Add family size as new feature.

        Returns:
            None.

        """
        self.train["FamilySize"] = self.train.SibSp + self.train.Parch + 1
        self.test["FamilySize"] = self.test.SibSp + self.test.Parch + 1
        
    def _add_fare_per_person(self):
        """
        Split fare by number of people in group purchase and add new fare
        as new feature.

        Returns:
            None.

        """
        self.train['FarePerPerson'] = self.train.Fare / self.train.FamilySize
        self.test['FarePerPerson'] = self.test.Fare / self.test.FamilySize
       
    def _add_age_segments(self):
        """
        Segment age into group of 10 years and add new age segments 
        as new feature.

        Returns:
            None.

        """
        seg = range(0, self.train.Age.max().astype(int)+1, 10)
        self.train['AgeSeg'] = pd.cut(self.train.Age, seg, 
                                      labels=np.arange(len(seg)-1))
        self.test['AgeSeg'] = pd.cut(self.test.Age, seg, 
                                     labels=np.arange(len(seg)-1))
        
    def _extract_title(self, df):
        """
        Extract title in Name as new feature.

        Args:
            df (DataFrame): dataframe to be worked on.

        Returns:
            None.

        """
        titles = []
        for name in df.Name.str.split():
            for subs in name:
                if subs[-1] == ".":
                    titles.append(subs)
                    break
        df['Title'] = titles
    
    def _encode_title(self, df):
        """
        Group and label titles.

        Args:
            df (DataFrame): dataframe to be worked on.

        Returns:
            None.

        """
        
        """
        Title dictionary:
            Married or elder women = 1,
            Unmarried women or girls = 2,
            Boys = 3,
            Men = 4
            Men with special occupation = 5
        """ 
        title_dict = {'Lady.':1, 'Countess.':1, 'Mrs.':1, 'Mme.':1,
                      'Miss.':2, 'Mlle.':2, 'Ms.':2,
                      'Master.':3,     
                      'Mr.':4, 'Dr.':4, 'Sir.':4, 'Jonkheer.':4, 'Don.':4,
                      'Major.':5, 'Col.':5, 'Capt.':5, 'Rev.':5}
        df['TitleGroup'] = df.Title.map(title_dict)
        
        # fill null with 1 if passenger is female, 4 if male
        if df.query('(TitleGroup.isnull()) and (Sex==1)').index.any():
            women = df.query('(TitleGroup.isnull()) and (Sex==1)').index.values
            df.loc[women, 'TitleGroup'] = 1        
        
        elif df.query('(TitleGroup.isnull()) and (Sex==0)').index.any():
            men = df.query('(TitleGroup.isnull()) and (Sex==0)').index.values
            df.loc[men, 'TitleGroup'] = 1
        
        # Exception: a women doctor should be assigned to 1
        if df.query('(Sex==1) and (Title=="Dr.")').index.any():
            women_dr = df.query('(Sex==1) and (Title=="Dr.")').index.values
            df.loc[women_dr, 'TitleGroup'] = 1
    
    def _add_title_groups(self):
        """
        Add title groups as new feature.

        Returns:
            None.

        """
        self._extract_title(self.train)
        self._encode_title(self.train)
        self._extract_title(self.test)
        self._encode_title(self.test)
    
    def _add_group_stat(self, features, col_name):
        """
        Add the average survival rate of each combination of features 
        as new feature.

        Args:
            features (list): list of features.
            col_name (str): name for new feature.

        Returns:
            None.

        """
        group_stat_dict = dict(self.train.groupby(features).Survived.mean())
        
        # add new group statistics in training set
        self.train[col_name] = [group_stat_dict[tuple(group)] \
                                for group in self.train[features].values]
        
        # add new group statistics in test set
        avg = self.train[col_name].mean()
        values = []
        for group in self.test[features].values:
            if tuple(group) in group_stat_dict:
                values.append(group_stat_dict[tuple(group)])
            else:
                values.append(avg)
        self.test[col_name] = values
        self.test[col_name].fillna(self.test[col_name].mean(), inplace=True)
        
    def _add_group_stats(self):
        """
        Add new group statistics.

        Returns:
            None.

        """
        self._add_age_segments()
        self._add_group_stat(["Sex", "Pclass", "AgeSeg"], "SexClassAge")
        self._add_title_groups()
        self._add_group_stat(["Sex", "Pclass", "TitleGroup"], "SexClassTitle")
        self._add_group_stat(["Sex", "Pclass", "SibSp"], "SexClassSibSp")
        self._add_group_stat(["Sex", "Pclass", "Parch"], "SexClassParch")
        self._add_group_stat(["Sex", "Pclass", "Embarked"], "SexClassEmbark")
                
    def feature_engineering(self):
        """
        Add new engineered features.

        Returns:
            None.

        """
        self._extract_ticket_num()
        self._add_family_size()
        self._add_fare_per_person()
        self._add_group_stats()
    
    def _create_dummy_set(self):
        """
        Create dummy sets for feature selection.

        Returns:
            tmp_features (TYPE): dummy features.
            tmp_target (TYPE): dummy target.

        """
        obj_features = [feat for (feat, dtype) in \
                        dict(self.train.dtypes).items() if dtype=='O']
        tmp_features = self.train.drop(columns = obj_features)
        tmp_features = tmp_features.drop(columns = "Survived")
        tmp_target = self.train.Survived
        return tmp_features, tmp_target
        
    def add_feature_importance_matrix(self, model, col_name):
        """
        Train model with dummy set and add feature importances in matrix.

        Args:
            model: model with feature importances.
            col_name (str): name of column.

        Returns:
            None.

        """
        tmp_features, tmp_target = self._create_dummy_set()        
        model.fit(tmp_features, tmp_target)
        
        if hasattr(model, "feature_importances_"):
            self.feat_sel_matrix["feat_imp"] = model.feature_importances_
            self.feat_sel_matrix[col_name] = \
                self.feat_sel_matrix["feat_imp"].rank(ascending=False)
            del self.feat_sel_matrix["feat_imp"]
    
    def add_rfe_matrix(self, model):
        """
        Rank features using Recursive Feature Elimination (RFE) and add 
        feature ranking in matrix.

        Args:
            model: model for RFE.

        Returns:
            None.

        """
        tmp_features, tmp_target = self._create_dummy_train_set()
        model = GradientBoostingClassifier()
        rfe = RFE(model, n_features_to_select=1)
        score = rfe.fit(tmp_features, tmp_target)
        self.feat_sel_matrix["RFE"] = score.ranking_
        
    def feature_selection(self, n):
        """
        Examine feature select matrix and select n best features.

        Args:
            n (int): number in ranking.

        Returns:
            top_n_features (list): top n features.

        """
        self.feat_sel_matrix["Rank"] = self.feat_sel_matrix.sum(axis=1).rank()
        tmp_features, tmp_target = self._create_dummy_train_set()
        self.feat_sel_matrix.set_index(tmp_features.columns, inplace=True)
        is_top_n_features = self.feat_sel_matrix.Rank.le(n)
        top_n_features = list(self.feat_sel_matrix[is_top_n_features].index)
        
        return top_n_features
        
class Model:
    # model container
    
    def __init__(self, random_state):
        self.random_state = random_state 
        self.model_list = []
        self.fold = None
        self.best_acc = 0
        self.best_model = None
        self.grid = None
        
    def _load_model(self, model):
        """
        Load model and add attribute(s).

        Args:
            model: model reference.

        Returns:
            Model object.

        """
        if hasattr(model(), "random_state"):
            return model(random_state = self.random_state)
        return model()
        
    @property
    def models(self):
        """
        Model list.

        Returns:
            (list): list of models added to object.

        """
        return self.model_list

    @models.setter
    def models(self, model):
        self.model_list.append(self._load_model(model))
    
    @models.deleter
    def models(self):
        self.model_list = []
        
    def add_cv(self, n_splits):
        """
        Add stratified K-fold cross-validator.

        Args:
            n_splits (int): number of folds.

        Returns:
            None.

        """
        self.fold = StratifiedKFold(n_splits=n_splits)
    
    @property
    def best_score(self):
        """
        Best model with its accuracy.

        Returns:
            score (str): string containing best model and its accuracy.

        """
        score = "Best model:\nacc=%.6f\t%s\n" % \
            (self.best_acc, self.best_model)
        return score
    
    @best_score.setter
    def best_score(self, scores):
        acc = scores[0]
        model = scores[1]
        
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_model = model
    
    @best_score.deleter
    def best_score(self):
        self.best_acc = 0
        self.best_model = None
            
    def fit(self, train_feat, train_target, standardize=False):
        """
        Train model with data.

        Args:
            train_feat (DataFrame): feature columns in training set.
            train_target (DataFrame): target columns in training set.
            standardize (boolean, optional): should data be standardized and
            pipelined. Defaults to False.

        Returns:
            None.

        """
        for model in self.model_list:
            if standardize:
                model = Pipeline([("Scaler", StandardScaler()), 
                                  ("model", model)])
            scores = cross_val_score(model, train_feat, train_target, 
                                     cv = self.fold, scoring = 'accuracy')
            acc = np.mean(scores)
            Log.print_msg("INFO", "acc=%.6f\t%s" % (acc, model))
            
            # fine best model
            self.best_score = (acc, model)

        
    def grid_search(self, model, grid_params):
        """
        Configurate grid search with grid search parameters.

        Args:
            model: grid search estimator.
            grid_params (TYPE): parameters to grid search.

        Returns:
            None.

        """
        model = self._load_model(model)            
        self.grid = GridSearchCV(estimator = model, 
                                 param_grid = grid_params, 
                                 scoring = 'accuracy', 
                                 cv = self.fold)
    
    def grid_fit(self, train_feat, train_target):
        """
        Grid search.

        Args:
            train_feat (DataFrame): feature columns in training set.
            train_target (DataFrame): target columns in training set.

        Returns:
            None.

        """
        grid_result = self.grid.fit(train_feat, train_target)        
        best_acc = grid_result.best_score_
        model = self.grid.estimator
        best_model = model.set_params(**grid_result.best_params_)
        Log.print_msg("INFO", "acc=%.6f\t%s" % (best_acc, best_model))
        
        # find best model
        self.best_score = (best_acc, best_model)

    def predict_validation(self, train_feat, train_target,valid_feat, 
                           valid_target_true):
        """
        Evaluate model with validation set.

        Args:
            train_feat (DataFrame): feature columns in training set.
            train_target (DataFrame): target columns in training set.
            valid_feat (DataFrame): feature columns in validation set.
            valid_target_true (DataFrame): target columns in validation set.

        Returns:
            float: accuracy of validaiton set prediction.

        """
        self.best_model.fit(train_feat, train_target)    
        valid_target_pred = self.best_model.predict(valid_feat)
        return accuracy_score(valid_target_true, valid_target_pred)
    
    def predict_test(self, train_feat, train_target, test_feat, filename):
        """
        Predict Survived of test set.

        Args:
            train_feat (DataFrame): feature columns in training set.
            train_target (DataFrame): target columns in training set.
            test_feat (DataFrame): feature columns in test set.
            filename (str): filename with csv extension.

        Returns:
            None.

        """
        self.best_model.fit(train_feat, train_target)
        test_pred = self.best_model.predict(test_feat)        
        test_pred = pd.DataFrame(test_pred, 
                                 index=test_feat.index, 
                                 columns=["Survived"])
        test_pred.to_csv(filename)
    

class XgboostModel:
    def __init__(self):
        self.xgb_model = None
    
    def fit(self, train_feat, train_target, valid_size, params):
        """
        

        Args:
            train_feat (DataFrame): feature columns in training set.
            train_target (DataFrame): target column in training set.
            valid_size (float): the proportion of training set to be used for
            validation set
            params (dict): XGBoost hyperparameters.

        Returns:
            None.

        """
        x_train, x_valid, y_train, y_valid = train_test_split(train_feat, 
            train_target, test_size=valid_size, random_state=10)
            
        d_train = xgb.DMatrix(x_train, label=y_train)
        d_valid = xgb.DMatrix(x_valid, label=y_valid)        
        watchlist = [(d_train, 'train'), (d_valid, 'valid')]
        nrounds=10000
        
        self.xgb_model = xgb.train(params, d_train, nrounds, watchlist,
                                   early_stopping_rounds=400, maximize=True, 
                                   verbose_eval=50)
    
    def predict_test(self, test_feat, filename):
        """
        Predict Survived of test set.

        Args:
            test_feat (DataFrame): feature columns in test set.
            filename (str): filename with csv extension.

        Returns:
            None.

        """
        d_test = xgb.DMatrix(test_feat)
        test_pred = self.xgb_model.predict(d_test)
        test_pred = pd.DataFrame(np.where(test_pred > 0.5, 1, 0), 
                                 index=test_feat.index, 
                                 columns=["Survived"])
        test_pred.to_csv(filename)


def process_data(train, test, is_validation=False):
    """
    Data wrangling, data engineering, and feature selection.

    Args:
        train (DataFrame): training set..
        test (DataFrame): test or validation set.
        is_validation (boolean, optional): is test a validation set with 
        target feature, Survived, or test set without Survived. Defaults to False.

    Returns:
        train_features (DataFrame): feature columns in training set.
        train_target (DataFrame): target column in training set.
        valid_features (DataFrame): feature columns in validation set.
        valid_target (DataFrame): target column in validation set.
        test_features (DataFrame): feature columns in test set.

    """
    data = DataProcessing(train, test)
    data.preprocessing()
    data.feature_engineering()
    
    data.add_feature_importance_matrix(GradientBoostingClassifier(), "GBC")
    data.add_feature_importance_matrix(RandomForestClassifier(), "RFC")
    data.add_feature_importance_matrix(ExtraTreesClassifier(), "ETC")
    data.add_rfe_matrix(GradientBoostingClassifier())
    top_n_features = data.feature_selection(10)
    
    train_features = train[top_n_features]
    train_target = train.Survived
    
    if is_validation:
        valid_features = test[top_n_features]
        valid_target = test.Survived        
        return train_features, train_target, valid_features, valid_target
    else:
        test_features = test[top_n_features]
        return train_features, train_target, test_features
        

if __name__ == "__main__":
    # set log level
    Log.set_log_level("INFO")
    
    # load data
    train_df = pd.read_csv("train.csv", header=0, index_col=0)
    test_df = pd.read_csv("test.csv", header=0, index_col=0)
    
    # split validation set from training set
    validation = train_df.tail(round(len(train_df) * 0.1)).copy()
    train = train_df.iloc[:-len(validation)].copy()
    
    # process training set and validation set
    train_features, train_target, valid_features, valid_target = \
        process_data(train, validation, True)
    
    # create Model object and add models
    models = Model(10)
    models.add_cv(n_splits = 10)
    
    models.models = LogisticRegression
    models.models = LinearDiscriminantAnalysis
    models.models = KNeighborsClassifier
    models.models = GaussianNB
    models.models = SVC
    models.models = DecisionTreeClassifier
    models.models = GradientBoostingClassifier
    models.models = RandomForestClassifier
    models.models = ExtraTreesClassifier
    
    # train with vanilla models
    models.fit(train_features, train_target)
    print(models.best_score)
    
    # train with standardized data
    models.fit(train_features, train_target, True)
    print(models.best_score)
    
    # grid seach hyperparameters for GradientBoostingClassifier
    grid_params = dict(n_estimators = [50, 70, 90, 100],
                        criterion = ["friedman_mse", "mse"],
                        loss = ["deviance", "exponential"])
    models.grid_search(GradientBoostingClassifier, grid_params)
    models.grid_fit(train_features, train_target)
    print(models.best_score)
    
    # grid seach hyperparameters for RandomForestClassifier
    grid_params = dict(n_estimators=[10, 20, 30, 50],
                        criterion=["gini", "entropy"],
                        max_depth=[6, 8, 10])
    models.grid_search(RandomForestClassifier, grid_params)
    models.grid_fit(train_features, train_target)
    print(models.best_score)
    
    # predict validation set and show accuracy
    valid_pred = models.predict_validation(train_features, train_target,
                                            valid_features, valid_target)
    print("validation acc=%.6f" % valid_pred)
    
    # process test set
    train_features, train_target, test_features = \
        process_data(train_df, test_df)
    
    # predict test set and export to csv file
    models.predict_test(train_features, train_target, test_features, 
                        "test_pred.csv")
    
    # create XGBoost model
    xgb_model = XgboostModel()
    
    # define hyperparameters
    params = {'objective':'binary:logistic',
              'max_depth':15,
              'learning_rate':0.14,
              'eval_metric':'auc',
              'min_child_weight':1,
              'subsample':0.65,
              'colsample_bytree':0.4,
              'seed':29,
              'reg_lambda':2.8,
              'reg_alpha':0.09,
              'gamma':0,
              'scale_pos_weight':1,
              'n_estimators': 600,
              'nthread':-1}
    
    # train with xgboost
    xgb_model.fit(train_features, train_target, 0.1, params)
    
    # predict test set and export to csv file
    xgb_model.predict_test(test_features, "test_pred_xgb.csv")