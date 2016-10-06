cur_dir=/rap/jvb-000-aa/data/sarath/emnlp/bridge_caption_norm_trunc_copy
FILES=`ls $cur_dir/$1/output/valid_e2f_*`
for file in $FILES
do
        metric=`sh $cur_dir/eval.sh $file /rap/jvb-000-aa/data/sarath/emnlp/dccstor/cssblr/amrita/multilingual_captions/bridge_caption/data/final/fr/mscoco_maxlength16/test/cf_caption/captions_fr.`
        bleu_4=`echo $metric | grep -o "Bleu_4: [0-9.]* "`
        echo $file "  ::: "$bleu_4
done

#echo "VALID OVER ... PRINTING TEST"
#FILES=`ls $cur_dir/$1/output/valid_i2f_*`
#for file in $FILES
#do
#	echo $file
#        metric=`sh $cur_dir/eval.sh $file /rap/jvb-000-aa/data/sarath/emnlp/dccstor/cssblr/amrita/multilingual_captions/bridge_caption/data/final/fr/mscoco_maxlength16/test/cf_caption/captions_fr.`
#        bleu_4=`echo $metric | grep -o "Bleu_4: [0-9.]* "`
#        echo $file "  ::: "$bleu_4
#done

