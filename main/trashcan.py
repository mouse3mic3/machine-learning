
    if model_use_param_search:
      param_search_best_model = []
      param_search_best_param = []
      param_search_cv_result = []

      param_search_avg_test_score = []
      param_search_sd_test_score = []
      

      for idx, model in enumerate(param_search_created):
        print(f'\nFitting: {model}')
        model.fit(X = x_train, y = y_train)
        
        # Best
        param_search_best_model.append(model.best_estimator_)
        param_search_best_param.append(model.best_params_)
        param_search_cv_result.append(model.cv_results_)

        # Evaluation

          
    elif model_use_cv:
      model_cv_avg_test_score = []
      model_cv_sd_test_score = []
      model_cv_max_test_score = []
      model_cv_min_test_score = []

      model_cv_avg_fit_time = []
      model_cv_sd_fit_time = []
      model_cv_max_fit_time = []
      model_cv_min_fit_time = []

      for idx, model in enumerate(model_created):
        print(f'\nFitting: {model}')
        model_cv = cross_validate(estimator = model, cv = cv_created[idx], X = x_train, y = y_train, scoring = scoring_metrics_type)
        
        # Evaluation
        model_cv_avg_test_score.append(np.mean(model_cv['test_scores']))
        model_cv_sd_test_score.append(np.std(model_cv['test_scores']))
        model_cv_max_test_score.append(np,max(model_cv['test_scores']))
        model_cv_min_test_score.append(np.min(model_cv['test_scores']))

        model_cv_avg_fit_time.append(np.mean(model['fit_time']))
        model_cv_sd_fit_time.append(np.std(np.mean(model['fit_time'])))
        model_cv_max_fit_time.append(np.max(np.mean(model['fit_time'])))
        model_cv_min_fit_time.append(np.min(np.mean(model['fit_time'])))


    else:
      for idx, model in enumerate(model_created):
        print(f'\nFitting: {model}')
        model.fit(X = x_train, y = y_train)
    
  


    """ --- Finalization --- """
    if model_use_param_search:
      print(f'\nDone: returning fitted hyperparameter search instance. Please remember to refit using the best_estimators_ or best_params_.')
      model_result = pd.DataFrame({
        'Param Search Instance' : param_search_created,
        'Best Model' : param_search_best_model,
        'Best Param' : param_search_best_param,
        'CV Result' : param_search_cv_result,
      })
      return(param_search_created)
    
    elif model_use_cv:
      print(f'\nDone: returning CV result, using {scoring_metrics_type} as the scoring metrics.')
      model_result = pd.DataFrame({
        'Model Instance' : model_created,
        'Avg Score' : model_cv_avg_test_score,
        'Std Dev Score' : model_cv_sd_test_score,
        'Max Score' : model_cv_max_test_score,
        'Min Score' : model_cv_min_test_score,

        'Avg Time' : model_cv_avg_fit_time,
        'Std Dev Time' : model_cv_sd_fit_time,
        'Max Time' : model_cv_max_fit_time,
        'Min Time' : model_cv_min_fit_time
      })
      return(model_result)
    
    else:
      print('Done: returning model.')
      return()
  