load('ytaksi5.mat')
%d3kisa 3KM den düşük d3orta 3-5KM arası d3uzak 5KM üstü mesafeler 3. durak
%için

d3kisa = [];
d3orta = [];
d3uzak = [];
for i=1:verisayisi 
    if uzakliklar(i,3)<3000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d3kisa = [d3kisa ; T];
    elseif uzakliklar(i,3)<5000
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d3orta = [d3orta ; T];
    else
        B = Enlem(i);
        C = Boylam(i);
        T = [B C];
        d3uzak = [d3uzak ; T];
    end
end
d3kisa = array2table(d3kisa);
d3orta = array2table(d3orta);
d3uzak = array2table(d3uzak);