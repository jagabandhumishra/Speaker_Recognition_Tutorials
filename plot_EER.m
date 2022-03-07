function EER=plot_EER(true_scores,false_scores,c);


    [P_miss,P_fa] = Compute_DET(true_scores,false_scores);
  
     Set_DCF(10,1,.01);
   
    Set_DET_limits(0.01,.99,0.01,.99)
   
     Plot_DET (P_miss,P_fa,c,1);
  
%    [DCF_opt Popt_miss Popt_fa]=Min_DCF(P_miss,P_fa); 
   
%       diff=P_miss-P_fa;
   
      indices_eer= find(abs((P_fa-P_miss))<.01);
   
      P_eer=P_miss(indices_eer(1));
      
      % EER upto two decimal
      
%       P_eer=(round(P_eer*100))./100;
      
      % EER in percentage
      
      EER=(P_eer)*100;