Step 1: run ./script_to_run_de.sh if your target language is de (this will use the dataset released as part of the wmt16 workshop) or run ./script_to_run_fr.sh if your target language is fr (this will use mscoco dataset)

Step 2: You can then run the following to get the Bleu scores:

---------------------------------

For the configuration with Image,English,German:

for image 2 English captions:

cd ..

./eval.sh  im_en_01/output/valid_i2e_<step>.txt reference_data/en_val_en-de_r [to get the results on the validation set]

./eval.sh  im_en_01/output/test_i2e_<step>.txt reference_data/en_en-de_r [to get the results on the test set]

for image 2 German captions:

cd ..

./eval.sh  im_de_01/output/valid_i2e_<step>.txt reference_data/de_val_r [to get the results on the validation set]

./eval.sh  im_de_01/output/test_i2e_<step>.txt reference_data/de_r [to get the results on the test set]
-----------------------------------

For the configuration with Image,English,French:

for image 2 English captions:

cd ..

./eval.sh  im_en_01/output/valid_i2e_<step>.txt reference_data/en_val_en-fr_r [to get the results on the validation set]

./eval.sh  im_en_01/output/test_i2e_<step>.txt reference_data/en_en-fr_r [to get the results on the test set]

for image 2 French captions:

cd ..

./eval.sh  im_fr_01/output/valid_i2e_<step>.txt reference_data/fr_val_r [to get the results on the validation set]

./eval.sh  im_fr_01/output/test_i2e_<step>.txt reference_data/fr_r [to get the results on the test set]

Step 3: For the two-step model which uses the image2english caption generation and then uses a standard PBSMT model to translate the generated english captions to the target language (German or French), you need to collect the "Predicted:" sentences from im_en_01/output/test_i2e_<step>.txt and pass it to the PBSMT system trained on the desired language


