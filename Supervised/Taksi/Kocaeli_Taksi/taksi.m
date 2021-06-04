%işe yaramayan veriler atıldı tarih verisi haftanın kaçıncı günü ve kaçıncı
%saati oluğu belirtilecek  şekilde düzenlendi.

dataset = readtable('untitled.csv');

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






