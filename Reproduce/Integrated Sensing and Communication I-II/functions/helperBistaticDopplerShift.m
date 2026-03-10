function fd = helperBistaticDopplerShift(txPos, rxPos, tgtPos, tgtVel, fc)
% Doppler shift induced by a target located at tgtPos and moving with a
% velocity tgtVel on a signal transmitted by a static transmitter located
% at txPos and received by a static receiver located at rxPos

txPhi = atan2(txPos(2)-tgtPos(2, :), txPos(1) - tgtPos(1, :));
rxPhi = atan2(rxPos(2)-tgtPos(2, :), rxPos(1) - tgtPos(1, :));

c = physconst("LightSpeed");
fd = -fc * (tgtVel(1, :) .* (cos(txPhi) + cos(rxPhi)) + tgtVel(2, :) .* (sin(txPhi) + sin(rxPhi))) / c;

end