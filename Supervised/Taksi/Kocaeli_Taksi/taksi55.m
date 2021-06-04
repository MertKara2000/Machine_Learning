load('ytaksi5.mat')
%d5kisa 3KM den düşük d5orta 3-5KM arası d5uzak 5KM üstü mesafeler 5. durak
%için

d5kisa = [];
d5orta = [];
d5uzak = [];
for i=1:verisayisi 
    if uzakliklar(i,5)<3000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d5kisa = [d5kisa ; T];
    elseif uzakliklar(i,5)<5000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d5orta = [d5orta ; T];
    else
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d5uzak = [d5uzak ; T];
    end
end
d5kisa = array2table(d5kisa);
d5orta = array2table(d5orta);
d5uzak = array2table(d5uzak);