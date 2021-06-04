load('loc.mat')

for i=1:stime
    
    C = split(Time(i),' ')
    C(3) = []
    Time(i) = strcat(C(1 ),C(2))
end

