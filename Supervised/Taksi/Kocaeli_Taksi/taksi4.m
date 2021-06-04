% Tüm noktaların duraklara olan uzaklıkları belirlendi.Uzaklıklar adlı
% matrisin içine yerleştirildi.
load('ytaksi4.mat')
verisayisi = size(Tarih);
uzakliklar = zeros(verisayisi(1,1),Duraksayisi(1,1));

for i=1:Duraksayisi
    lat1 = Duraklar(i,1);
    lon1 = Duraklar(i,2);
    for j=1:verisayisi
        lat2 = Koordinatlar(j,1);
        lon2 = Koordinatlar(j,2);
        
        R = 6371; % Radius of earth in KM

        nLat = lat2 * pi / 180 - lat1 * pi / 180;
        nLon = lon2 * pi / 180 - lon1 * pi / 180;
        a = sin(nLat/2) * sin(nLat/2) + cos(lat1 * pi / 180) * cos(lat2 * pi / 180) * sin(nLon/2) * sin(nLon/2);
        c = 2 * atan2(sqrt(a), sqrt(1-a));
        d = R * c;
        uzakliklar(j,i) = d * 1000; % meters
        
    end    
end

