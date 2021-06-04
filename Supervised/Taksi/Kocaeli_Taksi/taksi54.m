load('ytaksi5.mat')
%d4kisa 3KM den düşük d4orta 3-5KM arası d4uzak 5KM üstü mesafeler 4. durak
%için

d4kisa = [];
d4orta = [];
d4uzak = [];
for i=1:verisayisi 
    if uzakliklar(i,4)<3000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d4kisa = [d4kisa ; T];
    elseif uzakliklar(i,4)<5000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d4orta = [d4orta ; T];
    else
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d4uzak = [d4uzak ; T];
    end
end
d4kisa = array2table(d4kisa);
d4orta = array2table(d4orta);
d4uzak = array2table(d4uzak);