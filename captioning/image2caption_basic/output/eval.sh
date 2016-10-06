FILES=`ls $1/output/valid_i2e_*`
for file in $FILES
do
	#echo $file
	cd ..
	metric=`./eval.sh output/$file /dccstor/cssblr/amrita/multilingual_captions/bridge_caption/data/final/fr/mscoco_maxlength16/valid/caption/valid_captions_$2.`
	bleu_4=`echo $metric | grep -o "Bleu_4: [0-9.]* "`
	echo $file "  ::: "$bleu_4
	cd output
done

echo "VALID OVER ... PRINTING TEST"
FILES=`ls $1/output/test_i2e_*`
for file in $FILES
do
        echo $file
	cd ..
        metric=`./eval.sh output/$file /dccstor/cssblr/amrita/multilingual_captions/bridge_caption/data/final/fr/mscoco_maxlength16/test/cf_caption/captions_$2.`
        bleu_4=`echo $metric | grep -o "Bleu_4: [0-9.]* "`
        echo $file "  ::: "$bleu_4
	cd output	
done
