function [X, Y, Z] = PyramidSurf(n, Scale, BaseCenter)

warning('off')

ng = pyramid_grid_size ( n );

pg = pyramid_unit_grid ( n, ng );
pg = pg.';


counter = 0;

PyrSurf = [];


for i = n:-1:0

    a = (n - i) / n;
    X = [-a; a; a; -a];
    Y = [-a; -a; a; a];

    Xg = pg(counter + (1:(n-i+1)^2), 1);
    Yg = pg(counter + (1:(n-i+1)^2), 2);
    Zg = pg(counter + (1:(n-i+1)^2), 3);



    [~,on] = inpolygon(Xg,Yg,X,Y);

    if i > 0

        PyrSurf = [PyrSurf; Xg(on) Yg(on) Zg(on)];

    else
        PyrSurf = [PyrSurf; Xg Yg Zg];

    end

    counter = pyramid_grid_size(n-i);

end

PyrSurf = Scale * PyrSurf + BaseCenter;

X = PyrSurf(:,1);
Y = PyrSurf(:,2);
Z = PyrSurf(:,3);

warning('on')

end

function value = pyramid_grid_size ( n )

  np1 = n + 1;
  value = ( np1 * ( np1 + 1 ) * ( 2 * np1 + 1 ) ) / 6;

end


function pg = pyramid_unit_grid ( n, ng )

  pg = zeros( 3, ng );

  g = 0;

  for k = n : - 1 : 0
    hi = n - k;
    lo = - hi;
    for j = lo : 2 : hi
      for i = lo : 2 : hi
        g = g + 1;
        pg(1,g) = i / n;
        pg(2,g) = j / n;
        pg(3,g) = k / n;
      end
    end
  end

end