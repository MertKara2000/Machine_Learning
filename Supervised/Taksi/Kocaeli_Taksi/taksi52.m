load('ytaksi5.mat')
%d2kisa 3KM den düşük d2orta 3-5KM arası d2uzak 5KM üstü mesafeler 2. durak
%için

d2kisa = [];
d2orta = [];
d2uzak = [];
for i=1:verisayisi 
    if uzakliklar(i,2)<3000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d2kisa = [d2kisa ; T];
    elseif uzakliklar(i,2)<5000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d2orta = [d2orta ; T];
    else
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d2uzak = [d2uzak ; T];
    end
end
d2kisa = array2table(d2kisa);
d2orta = array2table(d2orta);
d2uzak = array2table(d2uzak);
