clear;
files = dir('data/*.csv');

for file = files'
    label = strrep(file.name, '.csv', '');
    path = strcat('data/', file.name);
    M = csvread(path, 1, 1);
    features = strsplit(fgetl(fopen(path)), ',');
    features = features(2:end);
    Mt = transpose(M);
    sets(1).(label) = Mt;
end
dims = numel(features);
labels = fieldnames(sets);

for dim = 1 : dims
    co = get(gca, 'ColorOrder');
    figure;
    hold on;
    for i = 1:numel(labels)
        mat = sets.(labels{i});
        h = histfit(mat(dim,:));
        set(h(2), 'Color', co(i,:));
        set(h(2), 'LineWidth', 2);
        if(strcmp(labels{i}, 'training') || strcmp(labels{i}, 'test'))
            set(h(2), 'LineStyle', '--');
            set(h(2), 'LineWidth', 1)
        end
        delete(h(1));
    end
    xlabel('normalized range');
    ylabel('histogram curve estimation');
    legend(labels);
    t = title(features{dim});
    set(t,'Interpreter','none');
    
    % Save figures to file
    savefig(strcat('fig/', features{dim}))
    print(strcat('fig/', features{dim}), '-dpng');
    hold off;
end
