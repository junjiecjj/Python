function MyBoxSubPlotFunc(Data, Labels, Title,KSubPlots,Groups)


linewd                  = 0.8;
hcfontsize              = 20;
MarkerSize              = 9;

% ColorsArr = [0, 0.4470, 0.7410 ; 0.9 0.49 0 ; 0.9290    0.6940    0.1250];
ColorsArr = [0, 0.4470, 0.7410 ; 0.8500, 0.3250, 0.0980 ; 0.9290    0.6940    0.1250];
subplot = @(m,n,p) subtightplot (m, n, p, [0.01 0.05], [0.1 0.01], [0.1 0.01]);
% {LabelsML1(1:3),LabelsML2(1:3)}
if KSubPlots==3
    SPSize = size(Data,2)/3;    
    figure;
    subplot(1,3,1); hold on
    boxplot(10*log10(Data(:,1:SPSize)),'labels',LabelsFunc(Labels,1,SPSize), 'symbol', '','Colors',ColorsArr);
    ax1 = gca;
    set(findobj(gca,'type','line'),'linew',2)
    ylabel('[dB]');
    title(Title);
    set(gca, 'FontSize', hcfontsize/1.5);
    set(gca, 'LineWidth', linewd);
    box off ; grid on
    subplot(1,3,2); hold on
    boxplot(10*log10(Data(:,SPSize+1:2*SPSize)),'labels',LabelsFunc(Labels,SPSize+1,2*SPSize), 'symbol', '','Colors',ColorsArr);
    % boxplot(10*log10(DirectivitySSAll(:,4:6)),'labels',{LabelsML1(4:6)}, 'symbol', '');
    ax2 = gca;
    set(findobj(gca,'type','line'),'linew',2)
    title('Directivity - subspace' );
    set(gca, 'FontSize', hcfontsize/1.5);
    set(gca, 'LineWidth', linewd);
%     set(gca, 'YLim', get(ax1,'YLim'));
    % set(gca, 'YTick', get(ax1,'YTick'));
    box on ; grid on
    h = gca; h.YAxis.Visible = 'off';
    % hlink = linkprop([ax1 ax2],{'YTick','YLim'});
    subplot(1,3,3); hold on
    boxplot(10*log10(Data(:,2*SPSize+1:3*SPSize)),'labels',LabelsFunc(Labels,2*SPSize+1,3*SPSize), 'symbol', '','Colors',ColorsArr);
    ax3 = gca;
%     ylim([-15 11]); % this line should be applied only for the directivity figure of the ML method
%     ylim([-11 18]); % this line should be applied only for the SIR figure of the MVDR method
    set(findobj(gca,'type','line'),'linew',2)
    % title('Directivity - subspace' );
    set(gca, 'FontSize', hcfontsize/1.5);
    set(gca, 'LineWidth', linewd);
%     set(gca, 'YLim', get(ax1,'YLim'));
    box on ; grid on
    h = gca; h.YAxis.Visible = 'off';
    YLimUp      = max([get(ax1,'YLim') get(ax2,'YLim') get(ax3,'YLim')]);
    YLimDown    = min([get(ax1,'YLim') get(ax2,'YLim') get(ax3,'YLim')]);
    set(ax1, 'YLim', [YLimDown YLimUp]);
    set(ax2, 'YLim', [YLimDown YLimUp]);
    set(ax3, 'YLim', [YLimDown YLimUp]); 
     linkaxes([ax1,ax2,ax3],'y');
%     set(ax2, 'YLim', get(ax3,'YLim'));
%     set(ax1, 'YLim', get(ax3,'YLim'));
elseif KSubPlots==2
    if Groups==0
        Groups = [ size(Data,2)/2 ;size(Data,2)];    
    end    
    figure;
    subplot(1,2,1); hold on
%     boxplot(10*log10(Data(:,1:SPSize)),'labels',LabelsFunc(Labels,1,SPSize), 'symbol', '');
    boxplot(10*log10(Data(:,1: Groups(1))),'labels',LabelsFunc(Labels,1, Groups(1)), 'symbol', '','Colors',ColorsArr);
    ax1 = gca;
    set(findobj(gca,'type','line'),'linew',2)
    ylabel('[dB]');
    title(Title);
    set(gca, 'FontSize', hcfontsize/1.5);
    set(gca, 'LineWidth', linewd);
    box off ; grid on
    subplot(1,2,2); hold on
    boxplot(10*log10(Data(:, Groups(1)+1:sum(Groups))),'labels',LabelsFunc(Labels, Groups(1)+1,sum(Groups)), 'symbol', '','Colors',ColorsArr);
    % boxplot(10*log10(DirectivitySSAll(:,4:6)),'labels',{LabelsML1(4:6)}, 'symbol', '');
    ax2 = gca;
    set(findobj(gca,'type','line'),'linew',2)
    title('Directivity - subspace' );
    set(gca, 'FontSize', hcfontsize/1.5);
    set(gca, 'LineWidth', linewd);
%     set(gca, 'YLim', max(get(ax1,'YLim'),get(ax2,'YLim')));
%     if     get(ax1,'YLim') > get(ax2,'YLim')
%         set(ax2, 'YLim', get(ax1,'YLim'));
%     else
%         set(ax1, 'YLim', get(ax2,'YLim'));
%     end
    YLimUp      = max([get(ax1,'YLim') get(ax2,'YLim') ]);
    YLimDown    = min([get(ax1,'YLim') get(ax2,'YLim') ]);
    set(ax1, 'YLim', [YLimDown YLimUp]);
    set(ax2, 'YLim', [YLimDown YLimUp]);
    linkaxes([ax1,ax2],'y');
    % set(gca, 'YTick', get(ax1,'YTick'));
    box on ; grid on
    h = gca; h.YAxis.Visible = 'off';
elseif KSubPlots==1
    SPSize = size(Data,2)/2;
    figure;hold on    
    boxplot(10*log10(Data),'labels',Labels, 'symbol', '','Colors',[0, 0.4470, 0.7410]);
    ax1 = gca;
    set(findobj(gca,'type','line'),'linew',2)
    ylabel('[dB]');
    title(Title);
    set(gca, 'FontSize', hcfontsize/1.5);
    set(gca, 'LineWidth', linewd);
    box off ; grid on
%     subplot(1,2,2); hold on
%     boxplot(10*log10(Data(:,SPSize+1:2*SPSize)),'labels',LabelsFunc(Labels,SPSize+1,2*SPSize), 'symbol', '');
%     % boxplot(10*log10(DirectivitySSAll(:,4:6)),'labels',{LabelsML1(4:6)}, 'symbol', '');
%     ax2 = gca;
%     set(findobj(gca,'type','line'),'linew',2)
%     title('Directivity - subspace' );
%     set(gca, 'FontSize', hcfontsize/1.5);
%     set(gca, 'LineWidth', linewd);
%     set(gca, 'YLim', get(ax1,'YLim'));
%     % set(gca, 'YTick', get(ax1,'YTick'));
%     box on ; grid on
%     h = gca; h.YAxis.Visible = 'off';    
    
end
    function LabelsFinal = LabelsFunc(Labels,StartInd,EndInd)
        if length(Labels) ==2
            LabelsFinal = {Labels{1}(StartInd:EndInd),Labels{2}(StartInd:EndInd)};
        else
            LabelsFinal = {Labels{1}(StartInd:EndInd),Labels{2}(StartInd:EndInd),Labels{3}(StartInd:EndInd)};
        end
    end
end


