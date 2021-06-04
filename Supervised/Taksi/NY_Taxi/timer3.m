load('timer1.mat')

Time = cell2table(Time)

Time = categorical(Time.Time)
Latitude = table2array(Latitude)
Longitude = table2array(Longitude)

for i=1:stime
    Latitude(i) = Latitude(i)/10
    Longitude(i) = Longitude(i)/10
end