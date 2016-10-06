#source /dccstor/anirlaha1/deep/venv/bin/activate
#export LD_LIBRARY_PATH=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda/lib64:$LD_LIBRARY_PATH
#export CUDA_HOME=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda-7.5/
#export PYTHONPATH=$PYTHONPATH:/dccstor/anirlaha1/

lang=$1
mkdir "en_"$lang"_01"

#jbsub -queue p8 -require k80 -cores 1+1 -p "en_"$lang"_01" -err "en_"$lang"_01/err.txt" \
python ../train_transliteration.py --save_dir="en_"$lang"_01" --learning_rate=0.001 \
--batch_size=64 --rnn_size=1024 --embedding_size=1024 --valid_every=600 \
--max_epochs=50 \
--e_f_e_train='../datasets/clean/train/en-'$lang'.en' \
--e_f_f_train='../datasets/clean/train/en-'$lang'.'$lang \
--e_f_e_valid='../datasets/clean/valid/en-'$lang'.en' \
--e_f_f_valid='../datasets/clean/valid/en-'$lang'.'$lang \
--e_f_e_test='../datasets/clean/test/en-'$lang'.en' \
--e_f_f_test='../datasets/clean/test/en-'$lang'.'$lang

mkdir "en_"$lang"_02"

#jbsub -queue p8 -require k80 -cores 1+1 -p "en_"$lang"_02" -err "en_"$lang"_02/err.txt" \
python ../train_transliteration.py --save_dir="en_"$lang"_02" --learning_rate=0.001 \
--batch_size=64 --rnn_size=2048 --embedding_size=2048 --valid_every=600 \
--max_epochs=50 \
--e_f_e_train='../datasets/clean/train/en-'$lang'.en' \
--e_f_f_train='../datasets/clean/train/en-'$lang'.'$lang \
--e_f_e_valid='../datasets/clean/valid/en-'$lang'.en' \
--e_f_f_valid='../datasets/clean/valid/en-'$lang'.'$lang \
--e_f_e_test='../datasets/clean/test/en-'$lang'.en' \
--e_f_f_test='../datasets/clean/test/en-'$lang'.'$lang

mkdir "en_"$lang"_03"

#jbsub -queue p8 -require k80 -cores 1+1 -p "en_"$lang"_03" -err "en_"$lang"_03/err.txt" \
python ../train_transliteration.py --save_dir="en_"$lang"_03" --learning_rate=0.001 \
--batch_size=128 --rnn_size=1024 --embedding_size=1024 --valid_every=600 \
--max_epochs=50 \
--e_f_e_train='../datasets/clean/train/en-'$lang'.en' \
--e_f_f_train='../datasets/clean/train/en-'$lang'.'$lang \
--e_f_e_valid='../datasets/clean/valid/en-'$lang'.en' \
--e_f_f_valid='../datasets/clean/valid/en-'$lang'.'$lang \
--e_f_e_test='../datasets/clean/test/en-'$lang'.en' \
--e_f_f_test='../datasets/clean/test/en-'$lang'.'$lang

mkdir "en_"$lang"_04"

#jbsub -queue p8 -require k80 -cores 1+1 -p "en_"$lang"_04" -err "en_"$lang"_04/err.txt" \
python ../train_transliteration.py --save_dir="en_"$lang"_04" --learning_rate=0.001 \
--batch_size=128 --rnn_size=2048 --embedding_size=2048 --valid_every=600 \
--max_epochs=50 \
--e_f_e_train='../datasets/clean/train/en-'$lang'.en' \
--e_f_f_train='../datasets/clean/train/en-'$lang'.'$lang \
--e_f_e_valid='../datasets/clean/valid/en-'$lang'.en' \
--e_f_f_valid='../datasets/clean/valid/en-'$lang'.'$lang \
--e_f_e_test='../datasets/clean/test/en-'$lang'.en' \
--e_f_f_test='../datasets/clean/test/en-'$lang'.'$lang



mkdir $lang"_en_01"

#jbsub -queue p8 -require k80 -cores 1+1 -p $lang"_en_01" -err $lang"_en_01/err.txt" \
python ../train_transliteration.py --save_dir=$lang"_en_01" --learning_rate=0.001 \
--batch_size=64 --rnn_size=1024 --embedding_size=1024 --valid_every=600 \
--max_epochs=50 \
--e_f_e_train='../datasets/clean/train/en-'$lang'.'$lang \
--e_f_f_train='../datasets/clean/train/en-'$lang'.en' \
--e_f_e_valid='../datasets/clean/valid/en-'$lang'.'$lang \
--e_f_f_valid='../datasets/clean/valid/en-'$lang'.en' \
--e_f_e_test='../datasets/clean/test/en-'$lang'.'$lang \
--e_f_f_test='../datasets/clean/test/en-'$lang'.en'

mkdir $lang"_en_02"

#jbsub -queue p8 -require k80 -cores 1+1 -p $lang"_en_02" -err $lang"_en_02/err.txt" \
python ../train_transliteration.py --save_dir=$lang"_en_02" --learning_rate=0.001 \
--batch_size=64 --rnn_size=2048 --embedding_size=2048 --valid_every=600 \
--max_epochs=50 \
--e_f_e_train='../datasets/clean/train/en-'$lang'.'$lang \
--e_f_f_train='../datasets/clean/train/en-'$lang'.en' \
--e_f_e_valid='../datasets/clean/valid/en-'$lang'.'$lang \
--e_f_f_valid='../datasets/clean/valid/en-'$lang'.en' \
--e_f_e_test='../datasets/clean/test/en-'$lang'.'$lang \
--e_f_f_test='../datasets/clean/test/en-'$lang'.en'

mkdir $lang"_en_03"

#jbsub -queue p8 -require k80 -cores 1+1 -p $lang"_en_03" -err $lang"_en_03/err.txt" \
python ../train_transliteration.py --save_dir=$lang"_en_03" --learning_rate=0.001 \
--batch_size=128 --rnn_size=1024 --embedding_size=1024 --valid_every=600 \
--max_epochs=50 \
--e_f_e_train='../datasets/clean/train/en-'$lang'.'$lang \
--e_f_f_train='../datasets/clean/train/en-'$lang'.en' \
--e_f_e_valid='../datasets/clean/valid/en-'$lang'.'$lang \
--e_f_f_valid='../datasets/clean/valid/en-'$lang'.en' \
--e_f_e_test='../datasets/clean/test/en-'$lang'.'$lang \
--e_f_f_test='../datasets/clean/test/en-'$lang'.en'

mkdir $lang"_en_04"

#jbsub -queue p8 -require k80 -cores 1+1 -p $lang"_en_04" -err $lang"_en_04/err.txt" \
python ../train_transliteration.py --save_dir=$lang"_en_04" --learning_rate=0.001 \
--batch_size=128 --rnn_size=2048 --embedding_size=2048 --valid_every=600 \
--max_epochs=50 \
--e_f_e_train='../datasets/clean/train/en-'$lang'.'$lang \
--e_f_f_train='../datasets/clean/train/en-'$lang'.en' \
--e_f_e_valid='../datasets/clean/valid/en-'$lang'.'$lang \
--e_f_f_valid='../datasets/clean/valid/en-'$lang'.en' \
--e_f_e_test='../datasets/clean/test/en-'$lang'.'$lang \
--e_f_f_test='../datasets/clean/test/en-'$lang'.en'
