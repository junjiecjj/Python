function result_slc = slc(radarpara,ctrl,echo,echo_slc)

    num_slc = size(echo_slc,1);%对消波束个数
    
    %选干扰样本
    R_starts = 1:128:length(echo);
    Result = zeros(num_slc,length(R_starts)-1,1);
    for j = 1:num_slc
        for i = 1:length(R_starts)-1
            Rx = R_starts(i):R_starts(i+1)-1;
            Result(j,i) = abs( echo(1,Rx)*echo_slc(j,Rx)'  )./(norm(echo(1,Rx))*norm(echo_slc(j,Rx)));
        end
    end
    Result1 = max(Result,[],1);
    index_slc = find(Result1 > 0.5);               
    
    r = ctrl.Bomen_st + R_starts(1:end-1)*radarpara.Ts*1e6*150;
    r = r/1e3;
    figure;
    plot(r,Result,'.-');
    grid;
    xlabel('R/km');
    ylim([-1,2]);
    title('副瓣对消选样本');
    
    %计算权值
    W = 0;
    for i = 1:length(index_slc)
        Rx = R_starts(index_slc(i)):R_starts(index_slc(i)+1)-1;
        sample_echo = echo(1,Rx);
        sample_echo_slc = echo_slc(1,Rx);
        W  =  W + sum(sample_echo.*sample_echo_slc)/length(sample_echo) * inv(sum(sample_echo_slc.*sample_echo_slc)/length(sample_echo));
    end
    W = W/length(index_slc);

%     index =  13280;
%     W  =  echo(1,index)*echo_slc(1,index) * inv(echo_slc(1,index)*echo_slc(1,index));

    
    %副瓣对消
%     Ryy = echo_slc*echo_slc'/length(echo_slc);
%     Rxy = echo*echo_slc'/length(echo);
%     result_slc = echo - abs(Rxy)^2*inv(Rxy)/Ryy*echo_slc;
    result_slc = echo - W*echo_slc;

    t = (1:ctrl.N)*radarpara.Ts+ctrl.Bomen_st/150*1e-6;
    r = t*1e6*150;
    figure;
    plot(r/1e3,db(abs(result_slc)),'.-');
    grid;
    ylim([-20,80]);
end






