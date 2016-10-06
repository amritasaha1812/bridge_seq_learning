#!/bin/bash
lang1=$1
lang2=$2
FILES=$lang1"-en-"$lang2"*"
for model_number in $FILES
do
        if [[ -e $model_number"/output/e2f_acc.txt" ]];
        then
                best=`paste -d' ' $model_number"/output/e2f_acc.txt" $model_number"/output/g2f_acc.txt" | sort -k15,15 -t' ' -k30,30 -nr`
                #echo $best
                epoch_number=`echo $best | head -1 | cut -f3 -d" "`
                i=`echo $model_number| cut -f2 -d'_'`
                IFS='
		'
                set -f
                for line in $best;do
                        echo "Configuration :"$i" ::: "$line
                done
                set +f
                unset IFS       
        fi
done

