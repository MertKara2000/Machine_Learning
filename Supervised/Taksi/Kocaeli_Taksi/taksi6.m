%k means clustering algoritması uygulandı.
load('ytaksi6.mat')

d1kisa = table2array(d1kisa);
d1uzak = table2array(d1uzak);


X = d1uzak
k = 5

[idx,C] = kmeans(X,k)

figure;
plot(X(idx==1,1),X(idx==1,2),'r.','MarkerSize',12)
hold on
plot(X(idx==2,1),X(idx==2,2),'b.','MarkerSize',12)
hold on
plot(X(idx==3,1),X(idx==3,2),'g.','MarkerSize',12)
hold on
plot(X(idx==4,1),X(idx==4,2),'y.','MarkerSize',12)
hold on
plot(X(idx==5,1),X(idx==5,2),'c.','MarkerSize',12)
hold on
plot(X(idx==6,1),X(idx==6,2),'m.','MarkerSize',12)

%Cluster merkezlerinin görünmesini sağlar
%plot(C(:,1),C(:,2),'kx','MarkerSize',15,'LineWidth',3) 

legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Cluster 5','Centroids','Location','NW')
title 'Durak 1 Uzak Mesafeler Clustering'
hold off;

%Aşşağıdaki kod Clustering ile ayrılmış grupların ne kadar eşit dağıldığı 
%ile ilgili bize fikir verir.
%clust = kmeans(X,k);
%silhouette(X,clust)

