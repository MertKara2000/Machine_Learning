dataset = readtable('test.csv');
dataset(:,1) = []
dataset(:,6) = []

Xs = dataset(:,2);
Ys = dataset(:,3);
Time = dataset(:,1);

Time = table2array(Time)
stime= size(Time)

for i=1:stime
    
    C = split(Time(i),'-')
    Time(i) = C(3)
    
end

