dataset = readtable('test.csv');

Time = dataset(:,2)
Longitude = dataset(:,3)
Latitude = dataset(:,4)

stime= size(Time)
Time = table2array(Time)

for i=1:stime
    
    C = split(Time(i),' ')
    DayNumber = weekday(C(1))
    DayNumber = num2str(DayNumber)
    Time(i) = strcat(DayNumber,C(2))
    C = split(Time(i),':')
    Time(i) = C(1) 
     
end

