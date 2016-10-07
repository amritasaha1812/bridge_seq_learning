Step 1: 
run ./script_to_run.sh with the desired parameter settings. 

One of the crucial parameters is pretrain_epochs. Keeping it 20-30 epochs is advisable. The Other parameters like learning rate etc. has been hardcoded as default values in the train_bridge_captions.py code.

This will generate a directory (with name as specified by the save_dir parameter in the script_to_run.sh) which has the models stored every 600 steps (model-<steip_number> and model-<step_number>.meta) and the output predictions are inside the output directory inside save_dir. 

In that output directory valid_e2f_<step_number>.txt gives the true and predicted target sequence starting from the pivot input sequence (for that step in the training) of the validation set. And valid_i2f_<step_number>.txt gives the true and predicted target sequence starting from the image input (for that step in the training) of the validation set.

Step 2: From the current directory, run ./eval.sh <save_dir>/output/valid_i2f_<step_number>.txt f_r to get the final BLEU/Rouge/Meteor score of generating the target sequence from the image input (where <save_dir> is the name of the directory where the model is stored)

Step 3: Keep tracking the following:
        1. In the <save_dir>/output directory valid_i2e_<step_number>.txt gives the batchwise correlation obtained between image and pivot hidden representation
        2. In the <save_dir>/output directory valid_e2f_<setp_number>.txt gives the true and predicted value of the target sequence given the pivot sequence as input. At the end of the file the validation loss is printed as "l_e2f".
        3. It is important to keep in mind that the validation loss of decoding the target sequence from the pivot sequence has to be minimised while the correlation between the image and pivot hidden representation has to be maximised.

