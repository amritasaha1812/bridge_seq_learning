echo '---------------------------------------'
echo $1
grep "Predicted: " $1 | sed -e 's/Valid Predicted: //g' -e 's/Predicted: //g' -e 's/<EOS>//g' -e 's/<GO>//g' -e 's/<PAD>//g' -e 's/[ ]*$//g' > pred

~/anaconda/bin/python ../neuraltalk2-master/eval_CIDER/script.py $2'0' $2'1' $2'2' $2'3' $2'4' pred
~/anaconda/bin/python ../neuraltalk2-master/coco-caption/myeval.py pred.json ref.json
