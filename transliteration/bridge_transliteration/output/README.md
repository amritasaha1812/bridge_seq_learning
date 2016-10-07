Step1: for every language pair (where the source is a non-pivot language and the target is also another non-pivot language) run the ./script_to_run.sh command to create the models

the following hyperparameters are recommended for this setting
rnn_size, embedding_size, top_k = 1024, 2048 
batch_size = 64, 128
change_lr_after=2, 3, 4
reduced_lr= 0.0001, 0.0005 
lmbda=0.1, 05, 1

follow the example_parameter_tuning_experiments.sh to see an exhaustive list of parameters that were tried out on the bridge transliteration dataset

This will create a large set of folders for each language pair (named as <source_language>_<target_language>_<configuration_number> where <configuration number is what you had named that specific configuration when running script_to_run.sh)

Step 2: Submit this python script as a job on ur GPU (strongly recommended) and for each of the running configurations, track the validation results. For example if the current running configuration is named "ka-en-hi_01", you can see the validation results in ka-en-hi_01/output/e2f_acc.txt. This file will contain the validation results after every 600 epochs. Once you select the best row from this line you can take the corresponding number from ka-en-hi_01/output/g2f_acc.txt and put it in our results table.

You can monitor the job by doing the following:
	cd .. (if your current directory is "output")
	watch ./avg_cap_loss output/ka-en-hi_01/output/ 10

Step 3: The output predictions will be in ka-en-hi_01/output/valid_e2f_<step>.txt and ka-en-hi_01/output/valid_g2f_<step>.txt where,
	 --- the file "valid_e2f_<step>.txt" stores the validation results for the True and Predicted sequence in the target language when decoded from the pivot language at that training step
	 --- the file "valid_g2f_<step>.txt" stores the validation results for the True and Predicted sequence in the target language when decoded from the source non-pivot language at that training step

