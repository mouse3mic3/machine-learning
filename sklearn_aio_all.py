# Imports
from algorithm_list import *
import pandas as pd
import numpy as np
import sys

from datetime import datetime as dt

# Scikit-learn
from sklearn.model_selection import *
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.kernel_ridge import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.metrics import *
from sklearn.datasets import make_classification, make_regression


class the_preprocessor:
  def __init__():
    pass

class the_model:
  """
  -------------------- Class: the_model --------------------
  
  Use this class if you already have preprocessed dataset.

  This class provides methods to:
    - create model(s)
    - fit model(s)
    - cross validation
    - create submission files
  
  Note that each components are independent from one another
  (i.e you can use it just to do cross validation, etc) with
  various degree of user control.

  A. Create Model(s) with make_model

  B. Fit model with make_fit_model

  C. Create cross validation (CV) with make_cv

  D. Fit model + CV with make_fit_model

  E. Create submission files
  
  -----------------------------------------------------------
  """
  
  def __init__( self ):
    """
    -------------------- Method: Initialization --------------------
    
    Designed to store all arguments into instance's variable.

    If skip_init_check is False (default), it will also check
    initial values for unexpected data types.
    
    ----------------------------------------------------------------
    """

    self.estimator_algorithm_list = estimator_algorithm()

    # Allowable hyperparameter search type (param_search_type)
    self.param_search_type_allow = ['grid', 'randomized']

    # Path for submission
    self.path = '/Users/t-muhammad.zaki/Desktop/sklearn-aio/sklearn-aio/'

  def make_cv(self, 
              cv_type,
              n_splits = None,
              n_repeats = None,
              leave_p_out = None,
              test_size = None,
              random_state = 1000,
              **params
              ):
    
    if type(random_state) == type(None): random_state = np.random.randint(1, 99999)
    for key, value in  params.items():
      if key == 'cv_type': cv_type = value
      elif key == 'n_splits': n_splits = value
      elif key == 'n_repeats': n_repeats = value
      elif key == 'leave_p_out': leave_p_out = value
      elif key == 'test_size': test_size = value
      elif key == 'random_state': random_state = value
      else:
        print('Warning:', key, value, 'is not recognized.')

    try:
      if cv_type == 'kfold': instance = KFold(n_splits = n_splits, random_state = random_state)
      elif cv_type == 'repeated_kfold': instance = RepeatedKFold(n_splits, n_repeats, random_state)
      elif cv_type == 'loo': instance = LeaveOneOut()
      elif cv_type == 'lpp': instance = LeavePOut(leave_p_out)
      elif cv_type == 'shuffle_split': instance = ShuffleSplit(n_splits, test_size, random_state)
      elif cv_type == 'stratified_kfold' : instance = StratifiedKFold(n_splits)
      elif cv_type == 'stratified_shuffle_split' : instance = StratifiedShuffleSplit(n_splits, test_size, random_state)
      else:
        print(f'\nError! cv_type {cv_type} is not recognized.')
        raise

    except Exception:
      print(f'\nError! cv arguments for {cv_type} is not recognized.')
      raise(Exception)
    
    return(instance)
  
  def make_search(self,
                  search_type,
                  estimator = None,
                  search_space = None,
                  scoring = None,
                  cv = None,
                  n_iter = None,
                  random_state = None,
                  **params
                  ):
    for key, value in params.items():
      if key == 'search_type': search_type = value
      elif key == 'estimator': estimator = value
      elif key == 'search_space': search_space = value
      elif key == 'scoring': scoring = value
      elif key == 'cv': cv = value
      elif key == 'n_iter': n_iter = value
      elif key == 'random_state': random_state = value
      else:
        print('Warning:', key, value, 'is not recognized.')
    
    try:
      if search_type == 'grid': 
        instance = GridSearchCV(estimator, search_space, scoring = scoring, cv = cv)
      elif search_type == 'random': 
        instance = RandomizedSearchCV(estimator, search_space, n_iter = n_iter, scoring = scoring, cv = cv)
      else:
        print(f'\nError! search_type {search_type} is not recognized.')
        raise
    except Exception:
      print(f'\nError! search arguments for {search_type} is not recognized.')
      raise

    return(instance)

  def make_model (self,
                  # Dataset
                  use_fit = True,
                  x_train = None, y_train = None, x_test = None, y_test = None,

                  # Algorithm
                  estimator = None,

                  # Metrics
                  eval_metrics = None,
                  ):
    """
    -------------------- Method: Make ML Model --------------------
    
    Designed to create ML model(s).
    
    ---------------------------------------------------------------
    """
    """ --- Initial Check --- """
    # BACKLOG

    """ --- Create Model --- """

    # Define a variable to store model
    estimator_created = []
    cv_created = []
    search_created = []
    # Generate model instance
    for instance_algo, instance_detail in estimator['estimator'].items():
      try:
        est_detail = self.estimator_algorithm_list[instance_algo]
      except:
        print(f'\nEstimator {instance_algo} is not recognized.')
        raise
      
      try:
        instance_purpose = instance_detail['purpose']
        est_detail = est_detail[instance_purpose]
      except Exception:
        print('\nPurpose is not recognized')
        raise
        
      # Instance CV
      est_instance = est_detail['instance']
      if est_detail['require_cv']:
        try:
          est_cv = self.make_cv(**instance_detail['est_cv'])
          est_instance.set_params(cv = est_cv)
        except Exception:
          print(f'\nError: creating estimator CV.')
          raise
      
      # Apply Instance Params
      try:
        instance_param = instance_detail['est_param']
        est_instance.set_params(**instance_param)
      except Exception:
        print('Error: instance parameter is not recognized.')
        raise
      estimator_created.append(est_instance)


      """ --- Create Evaluation CV --- """
      # CV for evaluation
      eval_cv = None
      try:
        eval_cv = self.make_cv(**instance_detail['eval_cv'])
        print('Using', eval_cv, 'to evaluate', instance_algo)
      except:
        try:
          eval_cv = self.make_cv(**estimator['eval_cv'])
          print('Using global', eval_cv, 'to evaluate', instance_algo)
        except:
          print('Not using specified CV for evaluation.')
      
      if type(eval_cv) != type(None):
        cv_created.append(eval_cv)


      """ --- Create Hyperparameter Search --- """
      # Parameter Search 
      search = None
      try:
        search = self.make_search(estimator = est_instance, cv = eval_cv, **instance_detail['search'])
        search_created.append(search)
      except Exception:
        print('Not using parameter search for', instance_algo)

      if type(search) != type(None):
        search_created.append(search)
    # END OF FOR LOOP

    """ --- Fitting and Evaluating --- """
    eval_result = None
    # If not using search
    if type(search) != type(None):
      for model in estimator_created:
        # FIT
        try:
          model.fit(x_train, y_train)
        except:
          print('\nFit failed on', model)

        if type(x_test) != type(None) and type(y_test) != type(None):
          print('\nNot evaluating model on test data.')
        else:
          # EVAL NO CV
          #if not eval_cv:
          eval_result = self.make_eval(eval_metrics = eval_metrics, estimator = model,
                                    x_test = x_test, y_test = y_test)
          
          # EVAL CV 
          #else:
            
    else:
      for model in search_created:
        # FIT
        try:
          model.fit(x_train, y_train)
          if type(x_test) == type(None) and type(y_test) == type(None):
            print('\nNot evaluating model on test data.')
          else:
            # EVAL NO CV
            #if not eval_cv:
            eval_result = self.make_eval(eval_metrics = eval_metrics, estimator = model,
                                      x_test = x_test, y_test = y_test)
            
        except:
          print('\nFit failed on', model)

    result = {
      'Estimator' : estimator_created,
      'CV' : cv_created,
      'Search' : search_created,
      'Eval' : eval_result
    }
    return(result)
      
  
  def make_eval(self, 
                eval_metrics, 
                estimator, 
                x_test, y_test, 
                **params
                ):
    
    y_pred = None
    for key, value in params.items():
      if key == 'y_pred': y_pred = value

    if type(y_pred) != type(None):
      y_pred = estimator.predict(x_test, y_test)

    result = {}
    for metrics in eval_metrics:
      try:
        # Classifier
        if metrics == 'accuracy': instance = accuracy_score(y_true = y_test, y_pred = y_pred)
        elif metrics == 'f1': instance = f1_score(y_true = y_test, y_pred = y_pred)
        elif metrics == 'fbeta': instance = fbeta_score(y_true = y_test, y_pred = y_pred)
        elif metrics == 'precision': instance = precision_score(y_true = y_test, y_pred = y_pred)
        elif metrics == 'recall': instance = recall_score(y_true = y_test, y_pred = y_pred)

        # Regressor
        elif metrics == 'r2': instance = r2_score(y_true = y_test, y_pred = y_pred)
        elif metrics == 'mse': instance = mean_squared_error(y_true = y_test, y_pred = y_pred)
        elif metrics == 'mae': instance = mean_absolute_error(y_true = y_test, y_pred = y_pred)
        else:
          print(f'\nError! eval metrics {metrics} is not recognized.')
          instance = 0
      except Exception:
        print(f'\nError! metrics argument for {metrics} is not recognized.')
        instance = 0
      result[metrics] = instance
    
    result['_prediction'] = y_pred    
    return(result)


  def make_submission(self, model = None, x_test = None, test_id = None,
                      submission_model_name = None, submission_target_name = None, submission_use_header = False):

    """
    -------------------- Method: Make Submission --------------------
    
    Designed to create submission that is easily reproducible.
    
    -----------------------------------------------------------------
    """
    
    if type(model) == type(None):
      model = self.model

    if type(x_test) == type(None):
      x_test = self.x_test
    if type(x_test) == type(None) or not isinstance(x_test, pd.DataFrame) or not isinstance(x_test, np.ndarray):
      print('\nError: No test dataset is found or invalid dataset type. Please use dataframe or numpy array.')
      return 1

    if type(submission_model_name) == type(None):
      submission_model_name = self.submission_model_name

    if type(submission_target_name) == type(None):
      submission_target_name = self.submission_target_name
    
    if type(submission_use_header) == type(None):
      submission_use_header = self.submission_use_header
    
    if type(test_id) == type(None):
      test_id = self.test_id

    if type(test_id) == type(None) or not isinstance(test_id, pd.DataFrame):
        print('\nWarning: No test_id is found or invalid type (not dataframe). Only target column will be generated.')
        submission = pd.DataFrame({submission_model_name : model.predict(x_test)})
        submission.to_csv(f'{self.path}/result_{submission_model_name}_{dt.now()}.csv', index = False)

    else:
      test_id[submission_target_name] = model.predict(x_test)
      test_id.to_csv(f'{self.path}/result_{submission_model_name}_{dt.now()}.csv', index = False)
  


  