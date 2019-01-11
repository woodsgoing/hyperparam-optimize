# hyperparam-optimize
hyperparam opt framework

TPE(Tree-structed Parzen Estimate) based hyperparam optimization framework with predefined internal model warpper and hyperparam scope provide an easy way for best prediction.

In fast_hyper_opt, bayes_optimize() is the main public API, accept train datasheet, model function and hyperparam space to search the optimized hyperparam for best prediction.

In fast_model, int_module_***_space() and int_module_***() are predefined as internal modules for fast_hyper_opt/bayes_optimize(). You can self-define whatever model function and paired hyperparams as wish to pass to fast_hyper_opt/bayes_optimize().

To run fast_hyper_opt, just put fast_hyper_opt.py/fast_eda.py/fast_impute.py/fast_model.py together with your python code and import fast_hyper_opt. You can refer to test case, fast_hyper_opt_test to check these APIs, related test resource is attached. 

To run fast_hyper_opt_test, dont forget to update file_path as you environment.
