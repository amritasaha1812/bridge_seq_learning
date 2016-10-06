Step 1: run ./script_to_run_de.sh if your target language is de (this will use the dataset released as part of the wmt16 workshop) or run ./script_to_run_fr.sh if your target language is fr (this will use mscoco dataset)

Step 2: You can then run the following to get the Bleu scores:

for image 2 English captions:
cd ..
./eval.sh  im_en_01/output/valid_i2e_<step>.txt reference_data/en_val_r [to get the results on the validation set]
./eval.sh  im_en_01/output/test_i2e_<step>.txt reference_data/en_r

for image 2 German captions:
cd ..
./eval.sh  im_de_01/output/valid_i2e_<step>.txt reference_data/de_val_r [to get the results on the validation set]
./eval.sh  im_de_01/output/test_i2e_<step>.txt reference_data/de_r

for image 2 French captions:
cd ..
./eval.sh  im_fr_01/output/valid_i2e_<step>.txt reference_data/fr_val_r [to get the results on the validation set]
./eval.sh  im_fr_01/output/test_i2e_<step>.txt reference_data/fr_r

Step 3: You need to collect the "Predicted:" sentences from im_en_01/output/test_i2e_<step>.txt and pass it to the PBSMT system. 
