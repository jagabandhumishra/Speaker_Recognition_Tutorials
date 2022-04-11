#!/bin/bash
datapath='/home/owner1/lre_jaga/test_data/lre09_30_out'
workpath='/home/owner1/kaldi/egs/lre_exp'
cd $datapath
 find -iname *.wav|cut -d '.' -f 2,3| awk '{printf "%s%s\n","/home/owner1/lre_jaga/test_data/lre09_30_out",$1}' >$datapath/tmp1.txt #put the database path
find -iname *.wav|cut -d '/' -f 2 >$datapath/tmp2.txt


sed -e 's/english_american_30/AENG/g' -e 's/farsi_30/FARSI/g' -e 's/georgian_30/GEORG/g' -e 's/korean_30/KOREA/g' -e 's/mandarin_30/MANDN/g' -e 's/russian_30/Rusia/g' -e 's/spanish_30/Spanish/g' -e 's/cantonese_30/Cantonese_chinese/g' -e 's/turkish_30/Turki/g' -e 's/vietnamese_30/Vietnamese/g' -e 's/creole_haitian_30/CREOL/g' $datapath/tmp2.txt> tmp2     #required if name of test languagelabel folder is different from train, in my case it is different, otherwise you can comment it.



find -iname *.wav|cut -d '.' -f 2| cut -d '/' -f 3 >$datapath/tmp3

paste $datapath/tmp3 $datapath/tmp2 $datapath/tmp1.txt|sort> $datapath/Data_key.txt


rm -rf tmp*

#cp $workpath/Data_key.txt $datapath/
cd $workpath
local/make_custom_data_test.pl $datapath data/lre09_30 #copy the utils folder from sre16(v2) and local from lre07(v1)
#./run.sh
