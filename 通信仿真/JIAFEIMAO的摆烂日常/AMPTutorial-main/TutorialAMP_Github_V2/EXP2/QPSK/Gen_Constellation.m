function xo=Gen_Constellation(mod_size)

normal_scal = 1/sqrt((2/3)*(2^mod_size-1)) ;      %Normalization factor
sym = modem.qammod(2^mod_size);
sym.inputtype = 'integer';   %输入数据为整数类型
sym.symbolorder = 'gray';    %格雷码
xo = normal_scal * modulate(sym, 0:2^mod_size-1); %得到星座点集合

end