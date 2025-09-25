classdef MyClass
    
    properties
        bandwidth
        avgRate
        association
        chn % channel coefficient
        chnMagnitude
        maxPower
        numBS
        numUser
        numTxAnte
        numRxAnte
        numSlot
        noise
        algorithm
    end
    
    methods
        function obj = MyClass( bandwidth, numBS, numUser, noise, numSlot,...
                numTxAnte, numRxAnte, chn, chnMagnitude, maxPower, association, algorithm)
            obj.bandwidth = bandwidth;
            obj.numBS = numBS;
            obj.numUser = numUser;
            obj.noise = noise;
            obj.numSlot = numSlot;
            obj.numTxAnte = numTxAnte;
            obj.numRxAnte = numRxAnte;
            obj.chn = chn;
            obj.chnMagnitude = chnMagnitude;
            obj.maxPower = maxPower;
            obj.association = association;
            obj.algorithm = algorithm;
        end
        
        [ schedule, U, V ] = runAlgorithm(obj, weight)
        [ numSchedule, avgRate ] = computeAvgRate(obj)
        currentRate = computeCurrentRate(obj, schedule, U, V)
        currentSINR = computeCurrentSINR(obj, schedule, U, V)
    end
    
end

