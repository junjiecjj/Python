function [ instRate] = runAlgorithm( obj, weight, initV )

numIter = 11;   

switch obj.algorithm
    case 'FP'
        [ instRate ] = runFP(obj, weight, numIter, initV);
    case 'WMMSE'
        [ instRate ] = runWMMSE( obj, weight, numIter, initV );
    case 'Newton'
        [ instRate ] = runNewton( obj, weight, numIter, initV );
    case 'SCALE'
        [ instRate ] = runSCALE( obj, weight, numIter, initV );
    otherwise
        error('unknown algorithm')
end

end