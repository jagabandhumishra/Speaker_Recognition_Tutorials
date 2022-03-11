%This program is written to analyze the eer of original and replay datta

clc;
close all; clear all;

root_folder='D:/ORIGINAL/GMM_UBM_RMFCC';

load([root_folder '/GMM_UBM_result/ubm_male_256_rmfcc_Result/m_true_scores.mat'])
load([root_folder '/GMM_UBM_result/ubm_male_256_rmfcc_Result/m_false_scores.mat'])

m_false_scores=m_false_scores;m_true_scores=m_true_scores;


score=[m_false_scores;m_true_scores];

mc=max(score);
m_false_scores=m_false_scores/mc;
m_true_scores=m_true_scores/mc;



minp=min(min(m_false_scores),min(m_true_scores));
maxp=max(max(m_false_scores),max(m_true_scores));



range=[minp:.06:maxp];

n1=histc(m_true_scores,range);
n2=histc(m_false_scores,range);

y1=n1/length(m_true_scores);
y2=n2/length(m_false_scores);

xmin=min(range);
xmax=max(range);
ymin=min(min(y1),min(y2));
ymax=max(max(y1),max(y2))+0.05;


plot(range,y1,'b', 'LineWidth',2);
hold on;
plot(range,y2, 'g','LineWidth',2);
axis([xmin xmax ymin ymax])


qp=range(1):.00004:range(end);
x=range;


iy1=interp1(x,y1,qp);
iy2=interp1(x,y2,qp);



cy1=find(iy1>0);
cy2=find(iy2>0);
cy=intersect(cy1,cy2);
cy=sort(cy);
qp1=qp(cy);

y3=iy1(cy);
y4=iy2(cy);



diff=abs(y3-y4);
[mdiff p]=min(diff);



line([qp1(p),qp1(p)],[0,ymax],'Color','m','LineWidth',2)

legend('True Scores','False Scores','Threshold');

th1=qp1(p);
preth1=th1;


it=0;
i=1;

while(i==1)
    
   FA=(length(find(m_false_scores>th1))/length(m_false_scores))*100;
   FR=(length(find(m_true_scores<th1))/length(m_true_scores))*100; 
   
   tolerance=0.04;
   
   inc=0.0001;
   
   diff=abs(FA-FR);
   
   if(diff>tolerance)
       
       if(FA>FR)
           th1=th1+inc;
       else
           
           th1=th1-inc; 
       end
       
   else
       
       i=0;
       
   end
    it=it+1;


end

hold off;




