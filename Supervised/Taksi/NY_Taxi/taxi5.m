load('time3.mat')

Location = array2table(Location)
Time = array2table(Time)

Table = [Location,Time]
writetable(Table,'test2.csv');

dataset2 = readtable('test2.csv');

Time = dataset2(:,2);
Time = categorical(Time.Time)

Long = dataset(:,2);
Lat  = dataset(:,3);
Location = dataset2(:,1);

Location = categorical(Location.Location)

for i=1:stime
    
 
   
end


