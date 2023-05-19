import pandas as pd
import numpy as np

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
from sklearn.metrics import accuracy_score, f1_score

def estimator_algorithm():
  return({
    # +++++ Supervised +++++ #
    ##### LINEAR MODEL #####
    'linear' : {
      'regressor' : {
        'instance' : LinearRegression(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
      }
    },
    'ridge' : {
      'classifier' : {
        'instance' : RidgeClassifier(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
      },
      'regressor' : {
        'instance' : Ridge(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
      },
    },
    'lasso' : {
      'regressor' : {
        'instance' : Lasso(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
      }
    },
    'lasso_cv' : {
      'regressor' : {
        'instance' : LassoCV(),
        'type' : 'supervised',
        'require_cv' : True,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'lasso_lars_cv' : {
      'regressor' : {
        'instance' : LassoLarsCV(),
        'type' : 'supervised',
        'require_cv' : True,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'lasso_lars_ic' : {
      'regressor' : {
        'instance' : LassoLarsIC(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'multi_task_lasso' : {
      'regressor' : {
        'instance' : MultiTaskLasso(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'elastic_net' : {
      'regressor' : {
        'instance' : ElasticNet(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'elastic_net_cv' : {
      'regressor' : {
        'instance' : ElasticNetCV(),
        'type' : 'supervised',
        'require_cv' : True,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'multi_task_elastic_net' : {
      'regressor' : {
        'instance' : MultiTaskElasticNet(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'multi_task_elastic_net_cv' : {
      'regressor' : {
        'instance' : MultiTaskElasticNetCV(),
        'type' : 'supervised',
        'require_cv' : True,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'lars' : {
      'regressor' : {
        'instance' : Lars(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'lasso_lars' : {
      'regressor' : {
        'instance' : LassoLars(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'orthogonal_matching_pursuit' : {
      'regressor' : {
        'instance' : OrthogonalMatchingPursuit(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'orthogonal_matching_pursuit_cv' : {
      'regressor' : {
        'instance' : OrthogonalMatchingPursuitCV(),
        'type' : 'supervised',
        'require_cv' : True,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
    },
    'bayes_ridge' : {
      'regressor' : {
        'instance' : BayesianRidge(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'bayes_ard' : {
      'regressor' : {
        'instance' : ARDRegression(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'logistic' : {
      'classifier' : {
        'instance' : LogisticRegression(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'logistic_cv' : {
      'classifier' : {
        'instance' : LogisticRegressionCV(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'tweedie' : {
      'regressor' : {
        'instance' : TweedieRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'sgd' : {
      'classifier' : {
        'instance' : SGDClassifier(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : SGDRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
    },
    'perceptron' : {
      'classifier' : {
        'instance' : Perceptron(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
    },
    'passive_aggresive' : {
      'classifier' : {
        'instance' : PassiveAggressiveClassifier(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : PassiveAggressiveRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'ransac' : {
      'regressor' : {
        'instance' : RANSACRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'theil_sen' : {
      'regressor' : {
        'instance' : TheilSenRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'huber' : {
      'regressor' : {
        'instance' : HuberRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'quantile' : {
      'regressor' : {
        'instance' : QuantileRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },

    ##### KERNEL RIDGE #####
    'kernel_ridge' : {
      'regressor' : {
        'instance' : KernelRidge(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },

    ##### ENSEMBLE #####
    'bagging_all' : {
      'classifier' : {
        'instance' : BaggingClassifier(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : True,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : BaggingRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'random_forest' : {
      'classifier' : {
        'instance' : RandomForestClassifier(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : RandomForestRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'extra_trees' : {
      'classifier' : {
        'instance' : ExtraTreeClassifier(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : ExtraTreeRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'ada_boost' : {
      'classifier' : {
        'instance' : AdaBoostClassifier(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : AdaBoostRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'gradient_boost' : {
      'classifier' : {
        'instance' : GradientBoostingClassifier(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : GradientBoostingRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'hist_gradient_boost' : {
      'classifier' : {
        'instance' : HistGradientBoostingClassifier(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : HistGradientBoostingRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },

    ##### SUPPORT VECTOR MACHINES #####
    'svm' : {
      'classifier' : {
        'instance' : SVC(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : SVR(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },
    'nu_svm' : {
      'classifier' : {
        'instance' : NuSVC(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : NuSVR(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
    },
    'linear_svm' : {
      'classifier' : {
        'instance' : LinearSVC(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : LinearSVR(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },

    ##### NEAREST NEIGHBORS #####
    'nn' : {
      'classifier' : {
        'instance' : KNeighborsClassifier(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : KNeighborsRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'clustering' : {
        'instance' : NearestNeighbors(),
        'type' : 'unsupervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
    },
    'nearest_centroid' : {
      'classifier' : {
        'instance' : NearestCentroid(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      }
    },

    ##### NAIVE BAYESIAN #####
    'gaussian_nb' : {
      'classifier' : {
        'instance' : GaussianNB(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
    },
    'multinomial_nb' : {
      'classifier' : {
        'instance' : MultinomialNB(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
    },
    'complement_nb' : {
      'classifier' : {
        'instance' : ComplementNB(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
    },
    'bernoulli_nb' : {
      'classifier' : {
        'instance' : BernoulliNB(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
    },
    'categorical_nb' : {
      'classifier' : {
        'instance' : CategoricalNB(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
    },

    ##### TREE #####

    'decision_tree' : {
      'classifier' : {
        'instance' : DecisionTreeClassifier(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
      'regressor' : {
        'instance' : DecisionTreeRegressor(),
        'type' : 'supervised',
        'require_cv' : False,
        'require_estimator' : False,
        'voting' : False,
        'multi_label' : False,
        'usable' : False,
      },
    },
  })
