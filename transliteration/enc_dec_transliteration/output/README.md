Step 1: Run ./train_en-x_x-en.sh

this will create separate folders for *_en and en_* (where * refers to any of the non-pivot languages like ma,hi,ka,ta). 4 configurations are usually experimented with in this setting
rnn_size=embedding_size: 1024, 2048 [rnn_size should always be same as embedding_size]
batch_size = 64, 128
learning_rate = 0.001
these configurations will result in 4 folders for every lanuage (like hi_en_01, hi_en_02, hi_en_03, hi_en_04). For details of this, please check train_en-x_x-en.sh

Step 2: It is advisable to run the above command only on GPU. Once the job run the following to monitor the losses:
watch ./avg_cap_loss en_hi_01/output/ 10 [the last argument is the number of epochs for which it will print the avg loss. The suggested way is to first run only one hyperparameter setting and the run this watch command to see the movement of the loss. This will help in understanding the loss-development in each of the configurations better] 

To check the accuracy on the validation set do:
cat en_hi_01/output/e2f_acc.txt


Step 3: To run the best model [based on validation accuracy] on the test data do:

	Step 3a: Lets assume that the best model was the one in en_hi_01 setting after 6600 steps. While training the validation error is computed after every 600 steps and the model is also saved after every 600 steps. So the best model  (which is model-6600 and model-6600.meta) has to be manually copied into en_hi_01/best (directory needs to be created if it does not exist)

	Step 3b: Run ./test.sh en_hi_01 

Step 4: To run the two-way encoder decoder model (which encodes any source non-pivot language to the pivot language and then decodes the target non-pivot language from the pivot), follow the steps
	Step 4a: Run Step 3 on the source non-pivot language with the configuration being of the form *_en_<configuration-number> where * is the source
	Step 4b: Run Step 3 on the target non-pivot language with the configuration being of the form en_*_<configuration-number> where * is the target
	Step 4c: Follow example_run_two-stage.sh to run the script ./two-stage.sh <source-language> <source_configuration-number> <target-language> <target_configuration-number>. for e.g. if
		hi -> source language
		ka -> target language
		hi_en_02 -> best performing configuration for source to pivot
		en_ka_03 -> best performing configuration for pivot to target
		then run the script: ./two-stage.sh hi 2 ka 3
