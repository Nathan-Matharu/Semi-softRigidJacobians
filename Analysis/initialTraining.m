%clear all;
%% Constants
fingerLengths = [1,1.5,2,2.5,3,3.5,4,4.5];

numFingers = length(fingerLengths);
numInputs = 7;
inVals = 4;
outVals = 2;

gndX = [918.9654,816.8962,794.6936088,829.1348839,739.8234879,836.8668941,816.5539935, 1117.539599];
gndY = [567.6915,725.2122,599.9239502,597.5074008,657.3000987,640.4734852,582.1009028, 391.5463097];
vmaxY = [1524.017683,1630.715,1514.129143,1597.242042,1518.573873,1565.794107,1628.058191, 791.3602242];
hmaxX = [223.7893839,152.5423,150.0800755,112.9402521,123.2098947,161.0409815,60.71429729, 738.2390954];

distToX = 150;
distToY = 190;
%% Excursions for each activation set for each finger

excursionRaw = [[2.5932	1.4933	1.0331	0.3152
0.4395	0.3682	0.4625	1.4128
0	0.6788	0.1427	1.6314
0.8145	0	0	-0.3359
-0.0023	1.3185	0	-0.0023
0	0	0.4832	0
-0.0989	0	0	1.9719];
    [2.9844	1.2241	1.1528	0.2738
0.4004	0.3221	0.474	1.3438
-0.0138	0.6121	0.0828	1.8385
0.8376	0	0	-0.3682
0	1.1321	0	-0.0092
0	0	0.497	0
-0.3958	-0.0023	0	2.2089];
    [2.6875	0.9135	0.8307	0.1104
0.5983	0.1795	0.3474	1.1137
-0.0207	0.4648	0	1.4772
1.0769	0	0	-0.3797
0	1.0101	0	0
0	0	0.5706	0
-0.3451	0	0	1.6222];
    [1.9926	0.9158	0.6443	0.1864
0.428	0.283	0.3382	1.0308
-0.0092	0.497	0.0184	1.728
0.6857	0	0	-0.3106
0	1.07	0	-0.023
0	0	0.4257	-0.0023
-0.3843	0	0	2.0548];
    [1.7879	0.8629	0.7363	0.2186
0.4142	0.3198	0.4487	1.2793
-0.0644	0.497	0.0851	1.9765
0.711	-0.0046	0	-0.3359
0	0.9089	0	0
0	0	0.4556	0
0	0	0	2.3286];
    [1.8385	0.8813	0.3313	0.0299
0.4556	0.2462	0.2071	1.0377
-0.0805	0.4809	0	1.514
0.6903	0	0	-0.2853
0	1.099	0	-0.0046
0	0	0.3313	-0.0023
-0.5706	0	0	1.781];
    [0.9206	0.6098	0.2784	0.2692
0.3866	0.1818	0.1496	0.6719
0	0.3359	0	1.0124
0.4188	0	0	0
0	0.78	0	0
0	0	0.3451	0
-0.0989	0	-0.0092	1.1114]; 
[0	0	0	0
1.0953	0.8191	1.7994	0.474
0.4165	0.6742	0.4602	0.7087
0	0	0	0
1.1942	-0.1404	0.0046	-0.0621
0	0	0	0
-0.5568	1.353	0.306	0.214]];

excursions = zeros(numInputs, inVals, numFingers);
for i = 1:numFingers
    excursions(:,:,i) = excursionRaw((i-1)*numInputs+1:i*numInputs,:);
end

%% End point pixels for each activation set for each finger

epRaw = [[633.8326502	914.9313833
778.118032	1062.366599
825.2189951	1079.006116
732.3635097	1028.127291
737.0398755	1046.145901
776.034145	1073.185343
783.7559838	1073.975915];
    [672.3688369	1201.094438
679.5587811	1183.134894
617.0902238	1173.147943
556.40325	1129.045921
552.266634	1152.913343
576.8147681	1166.302795
809.9210453	1206.557562];
    [711.532262	1050.006738
647.8657802	1034.641506
677.6527166	1030.752991
583.0524733	974.6021743
601.9872088	1021.254593
630.203197	1031.720516
675.261503	1029.984747];
    [699.7176323	1125.501882
639.3820643	1104.924038
799.5235248	1116.754747
641.7167661	1088.626649
628.1581769	1099.914369
605.6430819	1099.712669
850.1396779	1125.200569];
    [518.0555052	1152.77918
500.7249311	1154.872231
718.6654644	1190.628051
450.9932293	1113.641085
467.0814575	1132.783539
479.4100494	1145.137884
481.0628841	1146.117712];
    [502.6379809	1094.056621
607.5492206	1196.226269
782.4396107	1227.827124
559.7503817	1151.960564
558.3174992	1168.773043
594.7681932	1187.191237
603.5666616	1193.270175];
    [148.9018815	741.9566164
165.0153595	807.587358
160.6604048	784.9443405
151.3067601	749.2579052
152.6506145	765.6880049
156.5239658	771.2519007
169.7112298	811.9781573];
    [992.1306973	660.757953
992.0155072	658.7463562
1170.883734	692.3837476
904.3682256	612.3627939
1060.294748	702.0648992
931.3027838	631.8308597
930.5279186	630.3975649]];

epPix = zeros(numInputs, outVals, numFingers);
for i = 1:numFingers
    epPix(:,:,i) = epRaw((i-1)*numInputs+1:i*numInputs,:); 
end

%% Information for the resting position locations in pixels
restingPixels = [770.0054702	1063.307148; %1cm
                588.2572567	1158.489664; %1.5cm
                643.9360087	1030.747186; %2cm
                641.1594024	1101.594728; %2.5cm
                498.1596167	1146.664351; %3cm
                504.1842092	1094.243785; %3.5cm
                159.0408108	787.194775;  %4cm
                946.1466951	642.4832218];  %Hard finger

%% determine X and Y coordinates of each excursion set for each finger

xy = zeros(numInputs, outVals, numFingers);

for a = 1:numFingers
    for b = 1:numInputs
        epX = epPix(b, 1, a);
        epY = epPix(b, 2, a);
        x = ((gndX(a)-epX)/(gndX(a)-hmaxX(a)))*distToX;
        y = ((epY-gndY(a))/(gndY(a)-vmaxY(a)))*distToY;
        if a == 8
            y = ((epY-gndY(a))/(gndY(a)-vmaxY(a)))*(distToY-27);
        end
        xy(b,1,a) = x;
        xy(b,2,a) = y;
    end
end
xy;
%% Calculate the X and Y coordinates of the endpoint at the resting position
restingXY = [0,0;
    0,0;
    0,0;
    0,0;
    0,0;
    0,0;
    0,0;];
for a = 1:numFingers
    restingXY(a,1) = ((gndX(a)-restingPixels(a,1))/(gndX(a)-hmaxX(a)))*distToX;
    restingXY(a,2) = ((restingPixels(a,2)-gndY(a))/(gndY(a)-vmaxY(a)))*distToY;
end
restingXY;
%% Calculate difference in distance from the reference position
difXY = zeros(numInputs, outVals, numFingers);
for a = 1:numFingers
    for b = 1:numInputs
        difXY(b,1, a) = xy(b, 1, a)-restingXY(a,1);
        difXY(b,2, a) = xy(b, 2, a)-restingXY(a,2);
    end
end
difXY;
%% linear regression
linRegress = zeros(inVals,outVals,numFingers);
for a = 1: numFingers
    linRegress(:,:,a) = excursions(1,:,a)\difXY(1,:,a);
end
linRegress;
%this should work up to here now
%% calculate expected points based of the linear regressions and the excursion values for each finger
yCalc = zeros(numInputs,outVals,numFingers);
for a=1:numFingers
    yCalc(:,:,a) = excursions(:,:,a)*linRegress(:,:,a);
end
yCalc;
%%
rtsq = zeros(numFingers,1);
for a=1:numFingers
    [R,P] = corrcoef(difXY(:,:,a), yCalc(:,:,a));
    rtsq(a) = R(2,1);
end
rtsq = (rtsq.^2)
%% plotting
ws = warning('off', 'all');
numRuns = 90;
npoints = 75;
xyplot = zeros(npoints*numRuns, outVals, numFingers);
excur = zeros(npoints*numRuns, inVals, numFingers);
%{
for i=1:numRuns
    for a=1:numFingers
        alpha = linRegress(:,:,a);
        for b=1:npoints
            xx = 0;
            yy = 0;
            for c=1:inVals
                excur((i-1)*numRuns + b,c,a) = rand*2.4 - 1.2;
                xx = xx + alpha(c,1)*excur((i-1)*numRuns + b,c,a);
                yy = yy + alpha(c,2)*excur((i-1)*numRuns + b,c,a);
            end
            xyplot(b,1,a) = xx;
            xyplot(b,2,a) = yy;
        end
    end
    xyplot;
    for a=1:numFingers
        figure(a)
        hold on
        x = sortrows(xyplot(:,1,a));
        y = sortrows(xyplot(:,2,a));
        plot(x,y);
        p = polyfit(x,y,1);
        yfit = polyval(p,x);
        plot(x,yfit);
        hold off
    end
end
%}
%{
for a=1:numFingers
    secPlot = a+numFingers;
    figure(secPlot)
    p = parallelplot(excur(:,:,a));
end
%}
warning(ws)
%%
abc = flipud(rtsq);
ll = fliplr(fingerLengths);
figure(30)
hold on
plot(ll, abc)
p0 = polyfit(ll,abc,1);
yfit0 = polyval(p0,ll);
plot(ll,yfit0);
ylabel("R^2")
xlabel("Bone length")
hold off

figure(31)
hold on
plot(ll, abc)
p1 = polyfit(ll,abc,2);
yfit1 = polyval(p1,ll);
plot(ll,yfit1);
title("Using only prescribed posture 1 for training")
ylabel("R^2")
xlabel("Bone length")
hold off