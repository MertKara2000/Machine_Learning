load('taxi_time.mat')

Xs = table2array(Xs)
Ys = table2array(Ys)

Location = Xs

temp = 0;
for i=1:stime
   temp = (Xs(i)*Xs(i)) + (Ys(i)*Ys(i)) 
   temp = sqrt(temp)
   Location(i) = temp
end
