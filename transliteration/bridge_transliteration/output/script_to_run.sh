#source /dccstor/anirlaha1/deep/venv/bin/activate
#export LD_LIBRARY_PATH=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda/lib64:$LD_LIBRARY_PATH
#export CUDA_HOME=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda-7.5/
#export PYTHONPATH=$PYTHONPATH:/dccstor/anirlaha1/

lang1=$1
lang2=$2
batch_size=$3
dimension=$4
change_after=$5
reduced_lr=$6
lmbda=$7
dir_number=$8
#learning_rate=0.001
dir=$lang1"-en-"$lang2"_0"$dir_number
#mv $dir old_output
mkdir $dir

#jbsub -queue p8 -require k80 -cores 1+1 -p $dir -err $dir"/err.txt" \
python ../train_bridge_captions.py \
--save_dir=$dir --learning_rate=0.001 --pretrain_epochs=300 --batch_size=$batch_size --rnn_size=$dimension --embedding_size=$dimension \
--top_k=$dimension --valid_every=600 --save_every=600 --max_epochs=150 --change_lr_after=$change_after --reduced_lr=$reduced_lr --lmbda=$lmbda \
--g_e_g_train="../datasets/clean/train/en-"$lang1"."$lang1 \
--g_e_e_train="../datasets/clean/train/en-"$lang1".en" \
--g_e_g_valid="../datasets/clean/test/en-"$lang1"."$lang1 \
--g_e_e_valid="../datasets/clean/test/en-"$lang1".en" \
--e_f_e_train="../datasets/clean/train/en-"$lang2".en" \
--e_f_f_train="../datasets/clean/train/en-"$lang2"."$lang2 \
--e_f_e_valid="../datasets/clean/test/en-"$lang2".en" \
--e_f_f_valid="../datasets/clean/test/en-"$lang2"."$lang2 \
--g_f_g_test="../datasets/clean/test/en-"$lang1"."$lang1 \
--g_f_f_test="../datasets/clean/test/en-"$lang2"."$lang2 
