load('de.mat')

Longitude = array2table(Longitude)
Latitude = array2table(Latitude)

Longitude = [Longitude,Latitude]

writetable(Longitude,'coordinate.csv');

dataset = readtable('coordinate.csv');


