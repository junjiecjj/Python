function sec = calSecFunc(t)
date = datenum(t);  %日期转化为序列值
sec = 86400*mod(fix(date)-2,7)+ 3600*hour(t) + 60*minute(t) + second(t);
end