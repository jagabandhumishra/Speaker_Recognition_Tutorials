#!/bin/bash
datapath='/home/owner1/lre_jaga/language_train_data/data'
workpath='/home/owner1/kaldi/egs/lre_exp'
cd $datapath
 find -iname *.wav|cut -d '.' -f 2,3| awk '{printf "%s%s\n","/home/owner1/lre_jaga/language_train_data/data",$1}' >$workpath/tmp1.txt #put the database path
find -iname *.wav|cut -d '/' -f 2 >$workpath/tmp2.txt
find -iname *.wav|cut -d '.' -f 2| cut -d '/' -f 3 >$workpath/tmp3
paste $workpath/tmp3 $workpath/tmp2.txt $workpath/tmp1.txt|sort> $workpath/Data_key.txt
rm -f $workpath/tmp1.txt $workpath/tmp2.txt $workpath/tmp3
cp $workpath/Data_key.txt $datapath/
cd $workpath
local/make_custom_data.pl $datapath data #copy the utils folder from sre16(v2) and local from lre07(v1)
#./run.sh
