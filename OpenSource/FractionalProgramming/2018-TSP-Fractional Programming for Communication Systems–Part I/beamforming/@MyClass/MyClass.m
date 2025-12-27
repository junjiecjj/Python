classdef MyClass
    
    properties
        bandwidth
        avgRate
        association
        chn % channel coefficient
        distPathLoss
        maxPower
        numBS
        numUser
        numTxAnte
        numRxAnte
        numSlot
        noise
        algorithm
        numTone
        initV % initial V
    end
    
    methods
        function obj = MyClass( bandwidth, numTone, numBS, numUser, noise, numSlot,...
                mimoPattern, chn, chnMagnitude, maxPower, association, algorithm, initV)
            obj.bandwidth = bandwidth;
            obj.numBS = numBS;
            obj.numUser = numUser;
            obj.noise = noise;
            obj.numSlot = numSlot;
            obj.numTxAnte = mimoPattern(1);
            obj.numRxAnte = mimoPattern(2);
            obj.chn = chn;
            obj.distPathLoss = chnMagnitude;
            obj.maxPower = maxPower;
            obj.association = association;
            obj.algorithm = algorithm;
            obj.numTone = numTone;
            obj.initV = initV;
        end
    end
    
end

