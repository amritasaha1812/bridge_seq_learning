source /dccstor/anirlaha1/deep/venv/bin/activate
export LD_LIBRARY_PATH=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda-7.5/
export PYTHONPATH=$PYTHONPATH:/dccstor/anirlaha1/

dir=$1
model_number=`ls $dir"/best" | head -1`
#echo $model_number
lang1=`echo $1 | cut -f1 -d_`
lang2=`echo $1 | cut -f2 -d_`
non_en=""
if [ "$lang1" == "en" ]
then
	non_en=$lang2
elif [ "$lang2" == "en" ]
then 	
	non_en=$lang1
fi
#echo $non_en
jbsub -queue p8 -require k80 -cores 1+1 -out $dir"/test_log."$lang1"-"$lang2 -err $dir"/test_err."$lang1"-"$lang2 python ../test_transliteration.py  --e_test_file="../datasets/clean/test/en-"$non_en"."$lang1 --f_true_file="../datasets/clean/test/en-"$non_en"."$lang2 --f_out_file=$dir"/"$lang1"_"$lang2".pred" --config=$dir"/args.json" --vocabulary_dir=$dir --model=$dir"/best/"$model_number

