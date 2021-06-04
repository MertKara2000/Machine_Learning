% Duraklar ve noktalar koordinat düzeyine indirildi ve birleştirildi
load('ytaksi3.mat')

Denlem = table2array(Denlem);
Dboylam = table2array(Dboylam);

Duraksayisi = size(Denlem);

for i=1:verisayisi
    Enlem(i) = Enlem(i)/1000000;
    Boylam(i) = Boylam(i)/1000000;
            
end
Koordinatlar = [Enlem,Boylam];

for i=1:Duraksayisi
    Denlem(i) = Denlem(i)/1000000;
    Dboylam(i) = Dboylam(i)/1000000;
            
end

Duraklar = [Denlem,Dboylam];



writematrix(Koordinatlar,'Koordinat.csv');
