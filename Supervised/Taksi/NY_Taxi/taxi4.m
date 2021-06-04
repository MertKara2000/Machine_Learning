load('time2.mat')

for i=1:stime
    
    C = split(Time(i),':')
    C(3) = []
    Time(i) = C(1)
end
