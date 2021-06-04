load('timer1.mat')

Time = cell2table(Time)

Time = categorical(Time.Time)
Latitude = table2array(Latitude)

load('timer2.mat')
Latitude = num2cell(Latitude)

Latitude = cell2table(Latitude) 
Latitude = categorical(Latitude.Latitude)