% hierarchical clustering algritması uygulandı.
load('ytaksi6.mat')

d1kisa = table2array(d1kisa);
d1orta = table2array(d1orta);
d1uzak = table2array(d1uzak);

X = d1uzak;


Z = linkage(X,'ward');
dendrogram(Z);

T = cluster(Z,'cutoff',3,'Depth',4);
gscatter(X(:,1),X(:,2),T)

