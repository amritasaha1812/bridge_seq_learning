source /dccstor/anirlaha1/deep/venv/bin/activate
export LD_LIBRARY_PATH=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/dccstor/tgeorge5/software/PPC/INSTALLS/cuda-7.5/
export PYTHONPATH=$PYTHONPATH:/dccstor/anirlaha1/

lang1=$1
lang2=$3

dir_lang1=$lang1"_en_0"$2 
dir_lang2="en_"$lang2"_0"$4
echo $dir_lang1
echo $dir_lang2
two_stage_file=$dir_lang1"__"$dir_lang2
echo $two_stage_file
model_number=`ls $dir_lang2"/best" | head -1`
output_dir=$lang1"-"$lang2
echo -out $output_dir"/test_log."$two_stage_file -err $output_dir"/test_err."$two_stage_file python ../test_transliteration.py  --e_test_file=$dir_lang1"/"$lang1"_en.pred" --f_true_file="/dccstor/cssblr/mitesh/bridge_transliteration/datasets/clean/test/en-"$lang2"."$lang2 --f_out_file=$output_dir"/"$two_stage_file".pred" --config=$dir_lang2"/args.json" --vocabulary_dir=$dir_lang2 --model="/dccstor/cssblr/mitesh/enc_dec_transliteration/output/$dir_lang2/best/"$model_number
mkdir $output_dir
grep 'Predicted:' $dir_lang1"/"$lang1"_en.pred" | sed 's/^Predicted: //g' > $dir_lang1"/"$lang1"_en.pred_for_two-stage" 
jbsub -queue p8 -require k80 -cores 1+1 -out $output_dir"/test_log."$two_stage_file -err $output_dir"/test_err."$two_stage_file python ../test_transliteration.py  --e_test_file=$dir_lang1"/"$lang1"_en.pred_for_two-stage" --f_true_file="/dccstor/cssblr/mitesh/bridge_transliteration/datasets/clean/test/en-"$lang2"."$lang2 --f_out_file=$output_dir"/"$two_stage_file".pred" --config=$dir_lang2"/args.json" --vocabulary_dir=$dir_lang2 --model="/dccstor/cssblr/mitesh/enc_dec_transliteration/output/$dir_lang2/best/"$model_number
