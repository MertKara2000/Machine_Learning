load('ytaksi5.mat')
%d6kisa 3KM den düşük d6orta 3-5KM arası d6uzak 5KM üstü mesafeler 6. durak
%için

d6kisa = [];
d6orta = [];
d6uzak = [];
for i=1:verisayisi 
    if uzakliklar(i,6)<3000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d6kisa = [d6kisa ; T];
    elseif uzakliklar(i,6)<5000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d6orta = [d6orta ; T];
    else
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d6uzak = [d6uzak ; T];
    end
end
d6kisa = array2table(d6kisa);
d6orta = array2table(d6orta);
d6uzak = array2table(d6uzak);