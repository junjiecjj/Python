function [TargetList_new, FusionStart] = ...
                TargetClassification(TargetList_curr, ...
                                    TargetList_pre, ...
                                    FusionStart, ...
                                    CurrentFrame, ...
                                    NumTargetInTrack)
%     update TargetList through current TargetList and previous TargetList
% 
%     if CurrentFrame=2, since A has already know the rough info of B, we
% input TargetList_pre of this time the real value.
% 
%     if the difference between CurrentFrame and FusionStart is larger than
% a given threshold, the TargetList_pre is the previous fused result, or
% else we input TargetList_pre the previous measured results.
% 
%     if the # of TargetList_curr larger than TargetList_pre, this means
% the appearance of a new target. if smaller, this means the disappearance
% of one old target. if appear, FusionStart.C = CurrentFrame
% 
%     we don't consider there exists more than 2 targets in the
% environment, if TargetList_curr has more than 2 targets, we will delete
% the third target.
%%
if CurrentFrame == 2
    if 
end
end