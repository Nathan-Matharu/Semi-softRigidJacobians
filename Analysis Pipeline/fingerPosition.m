close all;
clearvars;
clc;
%%
files = dir('./Data/Deeplabcut_h5_Files');
files = files([files.isdir]==0);
bone = strings(1,length(files));
run = strings(1,length(files));
name = strings(1,length(files));
ID = strings(1,length(files));

%% Solved symbolic Endpoint kinematics
syms L1 L2 L3 L4 q1 q2 q3 q4
sims = [L1, L2, L3, L4, q1, q2, q3, q4];
% T = [cos(q3)*(cos(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4)) - sin(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1))) - sin(q3)*(cos(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1)) + sin(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4))), - cos(q3)*(cos(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1)) + sin(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4))) - sin(q3)*(cos(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4)) - sin(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1))), 0, L2*(cos(q1)*cos(q4) - sin(q1)*sin(q4)) + L4*(cos(q3)*(cos(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4)) - sin(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1))) - sin(q3)*(cos(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1)) + sin(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4)))) + L1*cos(q4) + L3*(cos(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4)) - sin(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1)));
%      cos(q3)*(cos(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1)) + sin(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4))) + sin(q3)*(cos(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4)) - sin(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1))),   cos(q3)*(cos(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4)) - sin(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1))) - sin(q3)*(cos(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1)) + sin(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4))), 0, L2*(cos(q1)*sin(q4) + cos(q4)*sin(q1)) + L4*(cos(q3)*(cos(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1)) + sin(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4))) + sin(q3)*(cos(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4)) - sin(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1)))) + L1*sin(q4) + L3*(cos(q2)*(cos(q1)*sin(q4) + cos(q4)*sin(q1)) + sin(q2)*(cos(q1)*cos(q4) - sin(q1)*sin(q4)));
%      0,                                                                                                                                                                                                           0, 1,                                                                                                                                                                                                                                                                                                                                                                     0;
%      0,                                                                                                                                                                                                           0, 0,                                                                                                                                                                                                                                                                                                                                                                     1];
L = [L1, L2, L3, L4];
q = [q1, q2, q3, q4];
% rotation about k by q4 Translation in j by l1, rotation about k by q1, translation in i by l2, rotation about k by q2, translation in i by l3, rotation about k by q3, translation in i by l4
T00 = [cos(q(4)),   -sin(q(4)),   0,  0;
       sin(q(4)),   cos(q(4)) ,   0,  0;
       0        ,   0         ,   1,  0;
       0        ,   0         ,   0,  1]; % rotated in k by q4

T01 = [1,   0,   0,  L(1);
       0,   1,   0,  0;
       0,   0,   1,  0;
       0,   0,   0,  1]; % Translation in i by -l1

T12 = [cos(q(1)),   -sin(q(1)),   0,  0;
       sin(q(1)),   cos(q(1)) ,   0,  0;
       0        ,   0         ,   1,  0;
       0        ,   0         ,   0,  1]; % rotated in k by -q1 CW is negative, right?

T23 = [1,   0,   0,  L(2);
       0,   1,   0,  0;
       0,   0,   1,  0;
       0,   0,   0,  1]; % Translation in i by l2

T34 = [cos(q(2)),   -sin(q(2)),   0,  0;
       sin(q(2)),   cos(q(2)) ,   0,  0;
       0        ,   0         ,   1,  0;
       0        ,   0         ,   0,  1]; % rotated in k by q2

T45 = [1,   0,   0,  L(3);
       0,   1,   0,  0;
       0,   0,   1,  0;
       0,   0,   0,  1]; % Translation in i by l3

T56 = [cos(q(3)),   -sin(q(3)),   0,  0;
       sin(q(3)),   cos(q(3)) ,   0,  0;
       0        ,   0         ,   1,  0;
       0        ,   0         ,   0,  1]; % rotated in k by q3

T67 = [1,   0,   0,  L(4);
       0,   1,   0,  0;
       0,   0,   1,  0;
       0,   0,   0,  1]; % Translation in i by l4 


TJ1 = T00*T01;
TJ2 = T00*T01 * T12 * T23;
TJ3 = T00*T01 * T12 * T23 * T34 * T45;
TEndpt = T00*T01 * T12 * T23 * T34 * T45 * T56 * T67;
GJ1 = simplify(TJ1(1:3,4));
GJ2 = simplify(TJ2(1:3,4));
GJ3 = simplify(TJ3(1:3,4));

G = [L4*cos(q1 + q2 + q3 + q4) + L2*cos(q1 + q4) + L1*cos(q4) + L3*cos(q1 + q2 + q4);
    L4*sin(q1 + q2 + q3 + q4) + L2*sin(q1 + q4) + L1*sin(q4) + L3*sin(q1 + q2 + q4);
                                                                              0];

%%
for i = 1:length(files)
    [bone{i}, run{i}] = getID(files(i));
    name(i) = strcat('./Data/Deeplabcut_h5_Files/',files(i).name);
    ID(i) = bone(i) + '_' + run(i);
end

[vtype(1:size(files))] = deal("cell");
Qcell = cell(1, length(files));  % Cell array to store Q tables
h5cell = cell(1, length(files));
%Qcell(1,:) = cellstr(ID);
for a = 1:length(files)
    h5Table = h5Data2Array(name(a));
    smoothH5Table = smoothData(h5Table);
    Qtable = getAngles(smoothH5Table,"deg"); % "deg" for deg
    Qcell{a} = Qtable;  % Store Q table in cell array
    Qtable = cell2table(Qcell,'VariableNames',ID);
    h5cell{a} = smoothH5Table;
    smoothH5Table = cell2table(h5cell,'VariableNames',ID);
end
predJ1 = zeros(3000,2,11);
predJ2 = zeros(3000,2,11);
predJ3 = zeros(3000,2,11);
predEndpt = zeros(3000,2,11);
Joint1 = zeros(3000,2,11);
Joint2 = zeros(3000,2,11);
Joint3 = zeros(3000,2,11);
Endpt = zeros(3000,2,11);
ErrorJ1 = zeros(3000,1,11);
ErrorJ2 = zeros(3000,1,11);
ErrorJ3 = zeros(3000,1,11);
ErrorEp = zeros(3000,1,11);

figure('Name',"Confidence Check")
for a = 1:length(files)
    miniHtable = smoothH5Table(:,a);
    miniHtable = miniHtable{1,1}{1,1}; % Get 2XXX x 11 table out of 1x1 table
    miniQtable = Qtable(:,a);
    miniQtable = miniQtable{1,1}{1,1};
    colums(a) = length(findPos(miniHtable,miniQtable,GJ1,sims));
    predJ1(1:colums(a),:,a) = findPos(miniHtable,miniQtable,GJ1,sims);
    predJ2(1:colums(a),:,a) = findPos(miniHtable,miniQtable,GJ2,sims);
    predJ3(1:colums(a),:,a) = findPos(miniHtable,miniQtable,GJ3,sims);
    predEndpt(1:colums(a),:,a) = findPos(miniHtable,miniQtable,G,sims);
    
    PosTable = miniHtable;
    Joint1(1:colums(a),:,a) = table2array([PosTable.Joint1(:,1),PosTable.Joint1(:,2)]);
    Joint2(1:colums(a),:,a) = table2array([PosTable.Joint2(:,1),PosTable.Joint2(:,2)]);
    Joint3(1:colums(a),:,a) = table2array([PosTable.Joint3(:,1),PosTable.Joint3(:,2)]);
    Endpt(1:colums(a),:,a) = table2array([PosTable.Endpoint(:,1),PosTable.Endpoint(:,2)]);

    ErrorJ1temp = Joint1(1:colums(a),:,a) - predJ1(1:colums(a),:,a);
    ErrorJ1(1:colums(a),:,a) = sqrt((ErrorJ1temp(:, 1) - ErrorJ1temp(:, 2)).^2);
    ErrorJ2temp = Joint2(1:colums(a),:,a) - predJ2(1:colums(a),:,a);
    ErrorJ2(1:colums(a),:,a) = sqrt((ErrorJ2temp(:, 1) - ErrorJ2temp(:, 2)).^2);
    ErrorJ3temp = Joint3(1:colums(a),:,a) - predJ3(1:colums(a),:,a);
    ErrorJ3(1:colums(a),:,a) = sqrt((ErrorJ3temp(:, 1) - ErrorJ3temp(:, 2)).^2);
    ErrorEptemp = Endpt(1:colums(a),:,a) - predEndpt(1:colums(a),:,a);
    ErrorEp(1:colums(a),:,a) = sqrt((ErrorEptemp(:, 1) - ErrorEptemp(:, 2)).^2);
    
    
    subplot(3,4,a)
    hold on
    scatter(1:colums(a),ErrorJ1(1:colums(a),:,a),'DisplayName','ErrorJ1')
    scatter(1:colums(a),ErrorJ2(1:colums(a),:,a),'DisplayName','ErrorJ2')
    scatter(1:colums(a),ErrorJ3(1:colums(a),:,a),'DisplayName','ErrorJ3')
    scatter(1:colums(a),ErrorEp(1:colums(a),:,a),'DisplayName','ErrorEndpt')
    ylabel("Kinematic Prediction Error (pixels)")
    xlabel("Time (recordings)")
    legend()
end

%%
%{
figure(1)
hold on
plot(1:length(Qtable.('4_4'){1}.Joint1),Qtable.('4_4'){1}.Joint1)
plot(1:length(Qtable.('4_4'){1}.Joint1),Qtable.('4_4'){1}.Joint2)
plot(1:length(Qtable.('4_4'){1}.Joint1),Qtable.('4_4'){1}.Joint3)
plot(1:length(Qtable.('4_4'){1}.Joint1),Qtable.('4_4'){1}.Ground)
%}

%% Functions

function [bone, run] = getID(file)
    % Returns the bonelength and run number from the filename
    name = file.name;
    index_ = strfind(name,'_');
    indexD = strfind(name,'DLC');
    bone = name(1:index_(1)-1);
    run = name(index_(1)+1:indexD-1);
end

function [h5Table,h5Array] = h5Data2Array(h5filename)
    % Returns a table or a string array of the data from the h5 file
    h5Data = h5read(h5filename,'/df_with_missing/table');
    index = double(h5Data.index);
    values_block_0 = h5Data.values_block_0.';
    bodyparts = ["Ground","Endpoint","hMax","vMax","Joint1","Joint2","Joint3","Bone1","Bone2","predEndpoint"];
    coords = ["x", "y", "likelihood"];
    title1 = strings(1,length(bodyparts)*3+1);
    title1(1) = "bodyparts";
    title2 = strings(1,length(bodyparts)*3+1);
    title2(1) = "coords";
    for i = 1:length(bodyparts)
        title1(i*3-1) = bodyparts(i);
        title1(i*3) = bodyparts(i);
        title1(i*3+1) = bodyparts(i);
        title2(i*3-1) = coords(1);
        title2(i*3) = coords(2);
        title2(i*3+1) = coords(3);
    end
    titles = [title1;title2];
    h5Array = [titles;index,values_block_0];
    Ground = table(str2double(h5Array(3:end,2)),str2double(h5Array(3:end,3)),str2double(h5Array(3:end,4)),'VariableNames',["x","y","likelihood"]);
    Endpoint = table(str2double(h5Array(3:end,5)),str2double(h5Array(3:end,6)),str2double(h5Array(3:end,7)),'VariableNames',["x","y","likelihood"]);
    hMax = table(str2double(h5Array(3:end,8)),str2double(h5Array(3:end,9)),str2double(h5Array(3:end,10)),'VariableNames',["x","y","likelihood"]);
    vMax = table(str2double(h5Array(3:end,11)),str2double(h5Array(3:end,12)),str2double(h5Array(3:end,13)),'VariableNames',["x","y","likelihood"]);
    Joint1 = table(str2double(h5Array(3:end,14)),str2double(h5Array(3:end,15)),str2double(h5Array(3:end,16)),'VariableNames',["x","y","likelihood"]);
    Joint2 = table(str2double(h5Array(3:end,17)),str2double(h5Array(3:end,18)),str2double(h5Array(3:end,19)),'VariableNames',["x","y","likelihood"]);
    Joint3 = table(str2double(h5Array(3:end,20)),str2double(h5Array(3:end,21)),str2double(h5Array(3:end,22)),'VariableNames',["x","y","likelihood"]);
    Bone1 = table(str2double(h5Array(3:end,23)),str2double(h5Array(3:end,24)),str2double(h5Array(3:end,25)),'VariableNames',["x","y","likelihood"]);
    Bone2 = table(str2double(h5Array(3:end,26)),str2double(h5Array(3:end,27)),str2double(h5Array(3:end,28)),'VariableNames',["x","y","likelihood"]);
    predEndpoint = table(str2double(h5Array(3:end,29)),str2double(h5Array(3:end,30)),str2double(h5Array(3:end,31)),'VariableNames',["x","y","likelihood"]);
    h5Table = table(index,Ground,Endpoint,hMax,vMax,Joint1,Joint2,Joint3,Bone1,Bone2,predEndpoint,'VariableNames',["index","Ground","Endpoint","hMax","vMax","Joint1","Joint2","Joint3","Bone1","Bone2","predEndpoint"]);
end


function slope = FindSlope(p1X, p1Y, p2X, p2Y)
    slope = -(p2Y - p1Y)/(p2X-p1X); % Function was used backwards up above
end

function [Qtable, angle] = getAngles(h5Table,unit)
    m = zeros(length(h5Table.index),4);
    angle = zeros(length(h5Table.index),4);
    % Pre-extraction for faster runtime
    EndpointX = h5Table.Endpoint.x;
    EndpointY = h5Table.Endpoint.y;
    Joint3X = h5Table.Joint3.x;
    Joint3Y = h5Table.Joint3.y;
    Joint2X = h5Table.Joint2.x;
    Joint2Y = h5Table.Joint2.y;
    Joint1X = h5Table.Joint1.x;
    Joint1Y = h5Table.Joint1.y;
    GroundX = h5Table.Ground.x;
    GroundY = h5Table.Ground.y;
    for b = 1:length(h5Table.index)
        % m = [endpt to joint3, joint3 to joint2, joint2 to joint1, joint 1 to ground]
        %m(b,:) = [FindSlope(h5Table.Endpoint.x(b),h5Table.Endpoint.y(b),h5Table.Joint3.x(b),h5Table.Joint3.y(b)), FindSlope(h5Table.Joint3.x(b),h5Table.Joint3.y(b),h5Table.Joint2.x(b),h5Table.Joint2.y(b)), FindSlope(h5Table.Joint2.x(b),h5Table.Joint2.y(b),h5Table.Joint1.x(b),h5Table.Joint1.y(b)),FindSlope(h5Table.Joint1.x(b),h5Table.Joint1.y(b),h5Table.Ground.x(b), h5Table.Ground.y(b))];
        m(b,:) = [FindSlope(EndpointX(b), EndpointY(b), Joint3X(b), Joint3Y(b)), FindSlope(Joint3X(b), Joint3Y(b), Joint2X(b), Joint2Y(b)), FindSlope(Joint2X(b), Joint2Y(b), Joint1X(b), Joint1Y(b)), FindSlope(Joint1X(b), Joint1Y(b), GroundX(b), GroundY(b))];
        % Q = [angle at joint 1, angle at joint 2, angle at joint 3, angle from ground to joint1]
        % Q(b,:,a) = [atan((m(a,1,b)-m(a,2,b))/(1+m(a,1,b)*m(a,2,b))), atan((m(a,3,b)-m(a,4,b))/(1+m(a,3,b)*m(a,4,b))), atan((m(a,5,b)-m(a,6,b))/(1+m(a,5,b)*m(a,6,b)))];
        if unit == "deg"
            %angle(b,:) = [atand((m(b,4)-0/(1+m(b,4)*0))), atand((m(b,1)-m(b,2))/(1+m(b,1)*m(b,2))), atand((m(b,2)-m(b,3))/(1+m(b,2)*m(b,3))), atand((m(b,3)-m(b,4))/(1+m(b,3)*m(b,4)))];
            angle(b,:) = -[atand((m(b,3)-m(b,4))/(1+m(b,3)*m(b,4))), atand((m(b,2)-m(b,3))/(1+m(b,2)*m(b,3))), atand((m(b,1)-m(b,2))/(1+m(b,1)*m(b,2))), atand((m(b,4)-0/(1+m(b,4)*0)))];
        else
            angle(b,:) = -[atan((m(b,3)-m(b,4))/(1+m(b,3)*m(b,4))), atan((m(b,2)-m(b,3))/(1+m(b,2)*m(b,3))), atan((m(b,1)-m(b,2))/(1+m(b,1)*m(b,2))), atan((m(b,4)-0/(1+m(b,4)*0)))];
        end
        %angle = -flip(angle,2); % [J1 angle, J2 angle, J3 angle, Grnd Angle];
        Q = angle;
        Qtable = table(Q(:,1),Q(:,2),Q(:,3),Q(:,4),'VariableNames',["Joint1", "Joint2", "Joint3", "Ground"]);
        if unit == "deg"       
            Qtable.Properties.VariableUnits = ["degrees","degrees","degrees","degrees"];
        else
            Qtable.Properties.VariableUnits = ["radians","radians","radians","radians"];
        end
    end
end

function predPos = findPos(PosTable,AngleTable,G,sims)
    [L1, L2, L3, L4, q1, q2, q3, q4] = deal(sims(1),sims(2),sims(3),sims(4),sims(5),sims(6),sims(7),sims(8));
    G_func = matlabFunction(G(1:2), 'Vars', [q1, q2, q3, q4, L1, L2, L3, L4]);
    ground = table2array([PosTable.Ground(:,1),PosTable.Ground(:,2)]);
    Joint1 = table2array([PosTable.Joint1(:,1),PosTable.Joint1(:,2)]);
    Joint2 = table2array([PosTable.Joint2(:,1),PosTable.Joint2(:,2)]);
    Joint3 = table2array([PosTable.Joint3(:,1),PosTable.Joint3(:,2)]);
    Endpt = table2array([PosTable.Endpoint(:,1),PosTable.Endpoint(:,2)]);
    for i = 1:height(PosTable)
        l = [pdist2(ground(i,:),Joint1(i,:)),pdist2(Joint1(i,:),Joint2(i,:)),pdist2(Joint2(i,:),Joint3(i,:)),pdist2(Joint3(i,:),Endpt(i,:))];
        if (AngleTable.Properties.VariableUnits{1} == 'degrees')
            newQ = deal([deg2rad(AngleTable.Joint1(i)), deg2rad(AngleTable.Joint2(i)), deg2rad(AngleTable.Joint3(i)), deg2rad(AngleTable.Ground(i))]);
        else
            newQ = deal([AngleTable.Joint1(i), AngleTable.Joint2(i), AngleTable.Joint3(i), AngleTable.Ground(i)]);
        end
        dist = double(G_func(newQ(1), newQ(2), newQ(3), newQ(4), l(1), l(2), l(3), l(4))); 
        predPos(i,:) = ground(i,:) - dist.';
    end
end

function smoothTable = smoothData(h5Table)
    k = 10; % Get k next values. Uncertainity most likely as finger is moving so more samples taken from future
    for i = 2:size(h5Table,2) % Check for each column
        column = h5Table{:,i};
        for j = 1:size(h5Table,1) % Check for each row
            if (column{j,3} <= 0.6) %Unconfident data point, needs to be smoothed
                weightedSum = 0;
                weights = zeros(1,k);
                % if (j < k/2+1)
                %     for a = 1:k
                %         weightedSum = weightedSum + column{j+a,1:2}*column{j+a,3}; %% add up the value
                %         weights(a) = column{j+b,3};
                %     end
                %     weightedMean = weightedSum / sum(weights);
                %     column{j,1:2} = weightedMean;
                % else
                %     for a = 1:k
                %         if mod(a, 2) == 0
                %             % If the index is even, add positive numbers
                %             b = ceil(a/2);
                %         else
                %             % If the index is odd, add negative numbers
                %             b = -ceil(a/2);
                %         end
                %         weightedSum = weightedSum + column{j+b,1:2}*column{j+b,3}; %% add up the value
                %         weights(a) = column{j+b,3};
                %     end
                %     weightedMean = weightedSum / sum(weights);
                %     column{j,1:2} = weightedMean;
                % end
                for a = 1:k
                    b = a-3; 
                    weightedSum = weightedSum + column{j+b,1:2}*column{j+b,3}; %% add up the value
                    weights(a) = column{j+b,3};
                end
                weightedMean = weightedSum / sum(weights);
                column{j,1:2} = weightedMean;
            end
        end
        h5Table{:,i} = column;
    end
    smoothTable = h5Table;
end