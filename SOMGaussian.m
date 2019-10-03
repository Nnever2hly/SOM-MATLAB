%in barname shabake asabi SOM baraye tabaghe bandi rang ast,dar in shabake
%az tabe,e gaussian estefade shode ast.bordar_vijegi=tedad vigegi 3 range
%RGB ast, neuron=tedad neuronkhoroji , epoch=tedad tekrar amuzesh be ezaye
%kole daade ha, vorudi=tedad vorudi baraye amuzesh shabake, eta0=nerkhe
%yadgiri avalie, zarib_eta=zarib kaheshe nerkhe yadgiri avalie,
%gau_variance=enheraaf me'yar gaussian, zarib_variance=zaribe kaheshe
%enheraf me'yar, namayesh=halathaye namayesh ke 1 be mani namayesh SOM
%tabaghe bandi rang va 2 be mani tabaghe bandi zende dar hengam amuzesh va
%tekrar amuzesh ast.
%
function som = SOMGaussian(bordar_vijegi, neuron, epoch, vorudi, eta0, zarib_eta, gau_variance, zarib_variance, namayesh)

radif = neuron;
sotun = neuron;

som =round(rand(radif,sotun,bordar_vijegi));

if namayesh >= 1
    fig = figure;
    namayeshSOM(fig, 1, 'SOM ba maghadir random', som, bordar_vijegi);
end

% tolide data amuzesh be surate random
train_Data = round(rand(vorudi,bordar_vijegi));

% tolid system mokhtasat
[x, y] = meshgrid(1:sotun,1:radif);

for t = 1:epoch    
    % mohasebe nerkhe yadgiri dar har tekrar
    eta = eta0 * exp(-t*zarib_eta);        

    
    %mohasebe enheraf me'yar taabe'e hamsayegi gaussian dar har tekrar
    sgm = gau_variance * exp(-t*zarib_variance);
    %arze taabe'e gaussian ra 3 sigma dar nazar migirim
    width = ceil(sgm*3);        
    
    for ntraining = 1:vorudi
        % daryaft e bordar amuzeshi dar har marhale
        trainingVector = train_Data(ntraining,:);
                
        % mohasebe fasele beine bordar amuzeshi va har neuron dar SOM
        fasele = mohasebe_fasele(trainingVector, som, radif, sotun, bordar_vijegi);
        
        % yaftane behtarin vahede motabeghat(bmu)
        [~, bmuindex] = min(fasele);
        
        % taghiir andise bmu be halate 2D
        [bmurow bmucol] = ind2sub([radif sotun],bmuindex);        
                
        % tolide taabe,e gaussian dar markaze mogheiate bmu
        g = exp(-(((x - bmucol).^2) + ((y - bmurow).^2)) / (2*sgm*sgm));
                        
        % ta'ine marze hamsayegi
        fromrow = max(1,bmurow - width);
        torow   = min(bmurow + width,radif);
        fromcol = max(1,bmucol - width);
        tocol   = min(bmucol + width,sotun);

        % daryafte neuron haye hamsaye va ta'ine size hamsayegi
        neuron_hamsaye = som(fromrow:torow,fromcol:tocol,:);
        sz = size(neuron_hamsaye);
        
        % taghiire bordare amuzeshi va taabe'e gaussian be halate chand bodi
        %baraye asan kardane mohasebe update vazn haye neuron
        
        T = reshape(repmat(trainingVector,sz(1)*sz(2),1),sz(1),sz(2),bordar_vijegi);                   
        G = repmat(g(fromrow:torow,fromcol:tocol),[1 1 bordar_vijegi]);
        
        % update vazn haye neuron ha ke dar hamsayegi bmu hastand
        neuron_hamsaye = neuron_hamsaye + eta .* G .* (T - neuron_hamsaye);

        % gharar dadane vazn haye jadide neuron haye hamsayegi bmu dar hameye SOM
        som(fromrow:torow,fromcol:tocol,:) = neuron_hamsaye;

        if namayesh == 2
            namayeshSOM(fig, 2, ['Epoch: ',num2str(t),'/',num2str(epoch),', Training Vector: ',num2str(ntraining),'/',num2str(vorudi)], som, bordar_vijegi);
        end        
    end
end

if namayesh == 1
    namayeshSOM(fig, 2, 'Trained SOM', som, bordar_vijegi);
    
end

function ed = mohasebe_fasele(trainingVector, sommap, radif, sotun, bordar_vijegi)

% tabdile namayeshe 3D neuron ha be 2D
neuronList = reshape(sommap,radif*sotun,bordar_vijegi);               

% Initialize Euclidean Distance
ed = 0;
for n = 1:size(neuronList,2)
    ed = ed + (trainingVector(n)-neuronList(:,n)).^2;
end
ed = sqrt(ed);

function namayeshSOM(fig, nsubplot, description, sommap, bordar_vijegi)
% namayeshe SOM tabaghe bandi range bedast amade

figure(fig);
subplot(1,2,nsubplot);
if bordar_vijegi >= 1
    imagesc(sommap(:,:,1:3));
else
    imagesc(sommap(:,:,1));
end
axis off;axis square;
title(description);
