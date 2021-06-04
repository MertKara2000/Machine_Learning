% tüm duraklar ve noktalar istenilen formata getirildi.
load('ytaksi1.mat')

Tarih = cell2table(Tarih);
Tarih = categorical(Tarih.Tarih);

% stable yi eklemezsen küçükten büyüğe sıralar
Denlem = unique(Denlem,'stable');
Dboylam = unique(Dboylam,'stable');

Duraklar = [Denlem,Dboylam];

Duraksayisi = size(Denlem);
Enlem = table2array(Enlem);
Boylam = table2array(Boylam);


for i=1:verisayisi
    for j=1:8
        if Enlem(i)<10000000
            Enlem(i)=Enlem(i)*10;
        end
    end
            
end

for i=1:verisayisi
    for j=1:8
        if Boylam(i)<10000000
            Boylam(i)=Boylam(i)*10;
        end
    end
            
end


%Taksi3 Buradan itibaren

Koordinatlar = [Enlem,Boylam]

for i=1:Duraksayisi
    EMin = Duraklar.Column19(i)-10000;
    EMax = Duraklar.Column19(i)+10000;
    BMin = Duraklar.Column20(i)-10000;
    BMax = Duraklar.Column20(i)+10000;
    for j=verisayisi:-1:1
        if Koordinatlar(j,1) > EMin && Koordinatlar(j,1) < EMax && Koordinatlar(j,2) > BMin && Koordinatlar(j,2) < BMax 
            Koordinatlar(j,:) = [];
            Tarih(j,:) = [];
            verisayisi(1,1) = verisayisi(1,1)-1;
           
        end
         
    end    
end

