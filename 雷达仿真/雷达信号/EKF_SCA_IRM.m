%% https://mp.weixin.qq.com/s?__biz=Mzk5MDU0NzkwNw==&mid=2247484051&idx=1&sn=21536a53da8885672256e551f8091f7c&chksm=c49f39fef0a0e32188643e2db5e0e995943528589c3d7ab99b6cb9aac7e8cd642b631678bcdd&mpshare=1&scene=1&srcid=0722wj9telovizeSCUhRBUml&sharer_shareinfo=0eed4252656a1b50d1ba43ac9fc46852&sharer_shareinfo_first=33826ade6d67e445eb41a5b0b4047739&exportkey=n_ChQIAhIQmvaBGq7kKPpfAehp3NtXXBKfAgIE97dBBAEAAAAAAKW2GeC2%2FgcAAAAOpnltbLcz9gKNyK89dVj0D3wLchUzOrnmIwfn58BhPrJxGLm2fLLuvRbvMt3siPTuETp%2BHfbcA2%2BS6G7gbhjsyaJvF0SVLrd0MS031v2tPElv5X0OoAh0SkoPcWj5uVfMIx8xOcm0Wmq1onMQZRxbBhLXEQ5IvarKXIt5qdYh3muwNaNQiAdtfzxPkxGGMyaOH1yT549KfsDG8D4vN%2FipPWBaiHLxuaKvAWS0cLOQ3fXxo4cjJh8TQEF54teMl4l5WNc8xFKlwlsvwT75%2B1Sh2KClSZGk0uveyPsptbz2y47no%2FXZugL8%2F6DFC1iTRrle4XiuWBuauWFlkeIdd7NclvtVZnARiGc2&acctmode=0&pass_ticket=x%2B%2FDVWtg9L6p65X2ArMCNaq5uldk6%2Bg3%2FYPViBSuCo3V8oNJyb6JQzJnD0YRPwNp&wx_header=0#rd

% 通感一体化中时序辅助波束成形技术文章及代码分享~

%SCA主循环：通过迭代凸逼近解决非凸优化问题
for l = 1: maxl
    W1dd = temp1;W2dd = temp2;
    %计算辅助变量B1和B2：对非凸项的一阶泰勒展开近似函数
    B1 = log(exp(1))/log(2)*(h1*h1')/(trace(h1*h1'*W2dd)+sigma0);   
    B2 = log(exp(1))/log(2)*(h2*h2')/(trace(h2*h2'*W1dd)+sigma0);
    %计算常数项a1dd和a2dd：泰勒展开的截距
    a1dd = log(trace(h1*h1'*W2dd)+sigma0)/log(2);
    a2dd = log(trace(h2*h2'*W1dd)+sigma0)/log(2);
%调用凸优化求解器CVX
cvx_solver Mosek_3
cvx_begin quiet
    %定义半正定矩阵
    variable W1opt(N,N) hermitian semidefinite
    variable W2opt(N,N) hermitian semidefinite
        rate1 = log(trace(h1*h1'*W1opt)+trace(h1*h1'*W2opt)+sigma0)/log(2) - real(a1dd + trace(B1*(W2opt - W2dd)));
        rate2 = log(trace(h2*h2'*W2opt)+trace(h2*h2'*W1opt)+sigma0)/log(2) - real(a2dd + trace(B2*(W1opt - W1dd)));
     %最小化负总速率（最大化总速率）
     minimize -rate1-rate2
    subject to
        %总功率约束     
        trace(W1opt+W2opt) == power;
           % 速率约束：确保每个用户的速率不低于全向通信的2倍
            rate1>=rateomni(1)*2;
            rate2>=rateomni(2)*2;
            %波束方向图约束（控制波束宽度）
            %对用户1的波束方向图约束
            for i= 0:(guji(1)-1)/2
                -0.05*trace(W1opt)<=(a(:,round((x_pre(1,1)*180/pi)*10+1+i))'*W1opt*a(:,round((x_pre(1,1)*180/pi)*10+1+i)))-(a(:,round((x_pre(1,1)*180/pi)*10+1))'*W1opt*a(:,round((x_pre(1,1)*180/pi)*10+1)))<=0.05*trace(W1opt);
                -0.05*trace(W1opt)<=(a(:,round((x_pre(1,1)*180/pi)*10+1-i))'*W1opt*a(:,round((x_pre(1,1)*180/pi)*10+1-i)))-(a(:,round((x_pre(1,1)*180/pi)*10+1))'*W1opt*a(:,round((x_pre(1,1)*180/pi)*10+1)))<=0.05*trace(W1opt);
            end
            %对用户2的波束方向图约束
            for i= 0:(guji(2)-1)/2
                -0.05*trace(W2opt)<=(a(:,round((x_pre(2,1)*180/pi)*10+1+i))'*W2opt*a(:,round((x_pre(2,1)*180/pi)*10+1+i)))-(a(:,round((x_pre(2,1)*180/pi)*10+1))'*W2opt*a(:,round((x_pre(2,1)*180/pi)*10+1)))<=0.05*trace(W2opt);
                -0.05*trace(W2opt)<=(a(:,round((x_pre(2,1)*180/pi)*10+1-i))'*W2opt*a(:,round((x_pre(2,1)*180/pi)*10+1-i)))-(a(:,round((x_pre(2,1)*180/pi)*10+1))'*W2opt*a(:,round((x_pre(2,1)*180/pi)*10+1)))<=0.05*trace(W2opt);
            end
cvx_end
%收敛判断：小于阈值时退出
if norm((W1opt - W1dd),1) < deltal
    break;
end
%更新迭代点
temp1 = W1opt;temp2 = W2opt;
l = l+1;
end




%IRM:对波束成形矩阵的秩1约束优化
for kk=1:maxk
    %逐步增大权重，强制解的低秩性
    w = w*wt;
    cvx_solver Mosek_3
    cvx_begin quiet
    %定义优化变量：半正定矩阵
    variable W1opt(N,N) hermitian semidefinite;
    variable W2opt(N,N) hermitian semidefinite;
    variable r nonnegative;
        rate1 = log(trace(h1*h1'*W1opt)+trace(h1*h1'*W2opt)+sigma0)/log(2) - real(a1dd + trace(B1*(W2opt - W2dd)));
        rate2 = log(trace(h2*h2'*W2opt)+trace(h2*h2'*W1opt)+sigma0)/log(2) - real(a2dd + trace(B2*(W1opt - W1dd)));
    %目标函数中加入秩惩罚项
    minimize -rate1*10000-rate2*10000+w*r
    subject to
        trace(W1opt+W2opt) == power;
            rate1>=rateomni(1)*2;
            rate2>=rateomni(2)*2;
            for i= 0:(guji(1)-1)/2
                -0.05*trace(W1opt)<=(a(:,round((x_pre(1,1)*180/pi)*10+1+i))'*W1opt*a(:,round((x_pre(1,1)*180/pi)*10+1+i)))-(a(:,round((x_pre(1,1)*180/pi)*10+1))'*W1opt*a(:,round((x_pre(1,1)*180/pi)*10+1)))<=0.05*trace(W1opt);
                -0.05*trace(W1opt)<=(a(:,round((x_pre(1,1)*180/pi)*10+1-i))'*W1opt*a(:,round((x_pre(1,1)*180/pi)*10+1-i)))-(a(:,round((x_pre(1,1)*180/pi)*10+1))'*W1opt*a(:,round((x_pre(1,1)*180/pi)*10+1)))<=0.05*trace(W1opt);
            end
            for i= 0:(guji(2)-1)/2
                -0.05*trace(W2opt)<=(a(:,round((x_pre(2,1)*180/pi)*10+1+i))'*W2opt*a(:,round((x_pre(2,1)*180/pi)*10+1+i)))-(a(:,round((x_pre(2,1)*180/pi)*10+1))'*W2opt*a(:,round((x_pre(2,1)*180/pi)*10+1)))<=0.05*trace(W2opt);
                -0.05*trace(W2opt)<=(a(:,round((x_pre(2,1)*180/pi)*10+1-i))'*W2opt*a(:,round((x_pre(2,1)*180/pi)*10+1-i)))-(a(:,round((x_pre(2,1)*180/pi)*10+1))'*W2opt*a(:,round((x_pre(2,1)*180/pi)*10+1)))<=0.05*trace(W2opt);
            end
        %秩1约束的松弛形式
        r*eye(N-1) - EV1(:,1:N-1)'*W1opt*EV1(:,1:N-1) == semidefinite(N-1);
        r*eye(N-1) - EV2(:,1:N-1)'*W2opt*EV2(:,1:N-1) == semidefinite(N-1);
    cvx_end

%更新特征向量
    [EV1,~] = eig(W1opt);
    [EV2,~] = eig(W2opt);
    fprintf('kk=%d,w = %f,r = %f\n',kk,w,r);
   %秩足够低时退出
   if r<deltak
        break;
    end


