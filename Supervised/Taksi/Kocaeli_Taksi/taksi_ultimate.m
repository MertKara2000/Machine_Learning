%işe yaramayan veriler atıldı tarih verisi haftanın kaçıncı günü ve kaçıncı
%saati oluğu belirtilecek  şekilde düzenlendi.

dataset = readtable('untitled.csv');

%Kaç metre için işlem yapacaksın?
ister = 4;
%Hangi durak için işlem yapacaksın
durakno = 3;



Tarih = dataset(:,1);
Enlem = dataset(:,2);
Boylam = dataset(:,3);
Denlem = dataset(:,6);
Dboylam = dataset(:,7);

verisayisi = size(Tarih);
Tarih = table2array(Tarih);

for i=1:verisayisi
    
    C = split(Tarih(i),' ');
    DayNumber = weekday(datenum(C(1),'dd.mm.yyyy'));
    DayNumber = num2str(DayNumber);
    Tarih(i) = strcat(DayNumber,C(2));
    C = split(Tarih(i),':');
    Tarih(i) = C(1) ;
     
end

Tarih = cell2table(Tarih);

% stable yi eklemezsen küçükten büyüğe sıralar
Denlem = unique(Denlem,'stable');
Dboylam = unique(Dboylam,'stable');
Denlem = table2array(Denlem);
Dboylam = table2array(Dboylam);
Duraksayisi = size(Denlem);
Enlem = table2array(Enlem);
Boylam = table2array(Boylam);

%Enlem ve Boylam Koordinat düzeyine düşürüldü
for i=1:verisayisi
   
    while Enlem(i)>100
        Enlem(i)=Enlem(i)/10;
    end 
end

for i=1:verisayisi
    while Boylam(i)>100
        Boylam(i)=Boylam(i)/10;
    end  
end
% Durakların enlem ve boylamları Koordinat düzeyine düşürüldü
for i=1:Duraksayisi
    while Dboylam(i)>100
       Dboylam(i)=Dboylam(i)/10;
    end 
end

for i=1:Duraksayisi
    while Denlem(i)>100
        Denlem(i)=Denlem(i)/10;
    end     
end

% Tüm noktalar ve durak noktaları birleştirildi.
Koordinatlar = [Enlem,Boylam];
Duraklar = [Denlem,Dboylam];
writematrix(Koordinatlar,'Koordinat.csv');

% Tüm noktaların duraklara olan uzaklıkları belirlendi.Uzaklıklar adlı
% matrisin içine yerleştirildi.

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

durak_mesafe = uzakliklar;

% Tüm noktaların tüm duraklara olan uzaklıkları 500 metreden başlayarak 500
% metre artarak 5000 metreye kadar alındı.mesafe_grubu adlı değişkene
% atandı.
for j=1:Duraksayisi
    mesafe = 0;
    for mesafe_grubu=1:10
        mesafe = mesafe+500;
        for i=1:verisayisi
            if uzakliklar(i,j)<mesafe && (mesafe-500) <uzakliklar(i,j)
                durak_mesafe(i,j) = mesafe_grubu;    
            end
        end
        
    end
end
% istenilen durak için istenilen mesafedeki noktalar ve bu noktaların tarih
% bilgisi diğerlerinden ayrıldı.
d1mesafe1 = [];
d1mesafe = [];

Tarih2 = [];
for sayi=1:10
    
    for i=1:verisayisi 
        if durak_mesafe(i,durakno) == ister && sayi==1
            B = Enlem(i);
            C = Boylam(i);
            T = [B C];
            d1mesafe1 = [d1mesafe1 ; T];
            Tarih2 = [Tarih2;Tarih(i,1)];
        end
        if durak_mesafe(i,durakno) == sayi
            B = Enlem(i);
            C = Boylam(i);
            T = [B C];
            d1mesafe = [d1mesafe ; T];
            
        end
    end
end

%Ayrılan noktalara hierarchical clustering uygulandı.
X = d1mesafe1;
Y = pdist(X);
Z = linkage(Y,'average');

%c = cophenet(Z,Y)
%dendrogram(Z,size(X,1));

T = cluster(Z,'maxclust',4);
gscatter(X(:,1),X(:,2),T);