load('ytaksi5.mat')
%d1kisa 3KM den düşük d1orta 3-5KM arası d1uzak 5KM üstü mesafeler 1. durak
%için

d1kisa = [];
d1orta = [];
d1uzak = [];
for i=1:verisayisi 
    if uzakliklar(i,1)<3000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d1kisa = [d1kisa ; T];
    elseif uzakliklar(i,1)<5000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d1orta = [d1orta ; T];
    else
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d1uzak = [d1uzak ; T];
    end
end
d1kisa = array2table(d1kisa);
d1orta = array2table(d1orta);
d1uzak = array2table(d1uzak);




