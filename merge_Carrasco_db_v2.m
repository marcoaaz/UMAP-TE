
%merge_Carrasco_db.m script

%This script documents the fusing process of the main and validation
%datasets from Carrasco-Godoy et al. 2024 paper into a single merged table
%that is studied in the Jupyter Notebooks provided with this m/s.

%Mismatching categorical variables have been homogenised within each column.

%Created: 12-Aug-24
%Updated: 22-Oct-25

%%
clear 
clc

%User input
workingDir = 'E:\Feb-March_2024_zircon imaging\05_zircon geochemical data\carrasco-godoy data\Zircon_Fertility_Data-main';
file1 = 'Zircon Fertility Data.csv'; %duplicated Y and Ba columns
file2 = 'External Validation.csv'; %only w/ Ca column

%Script

filepath1 = fullfile(workingDir, file1);
filepath2 = fullfile(workingDir, file2);
table_main = readtable(filepath1, 'VariableNamingRule','preserve');
table_external = readtable(filepath2, 'VariableNamingRule','preserve');

varNames1 = table_main.Properties.VariableNames;


%% Remove duplicate columns

table_main2 = table_main;
n_rows = size(table_main, 1);
duplicated_list = {'Y', 'Ba'};
n_duplicates = length(duplicated_list);
for i = 1:n_duplicates
    new_col = strings(n_rows, 1);

    item = duplicated_list{i};
    item2 = strcat(item, '_1'); %default
    item_idx = find(strcmp(item, varNames1));

    col_temp1 = table_main{:, item};
    col_temp2 = table_main{:, item2};
    col_temp = table_main(:, {item, item2});
    condition1 = iscell(col_temp1);
    condition2 = iscell(col_temp2);

    if condition1
        idx_nan1 = strcmp('NA', col_temp1);
    else        
        idx_nan1 = isnan(col_temp1);        
    end
    if condition2
        idx_nan2 = strcmp('NA', col_temp2);
    else        
        idx_nan2 = isnan(col_temp2);        
    end    
    idx_nan = idx_nan1 & idx_nan2;
    idx_array = [idx_nan1, idx_nan2];

    new_col(idx_nan, 1) = 'NA';
    for j = 1:n_rows
        idx_col = find(~idx_array(j, :), 1); %first only

        condition3 = isempty(idx_col);
        if condition3
            new_col(j, :) = 'NA';
        else
            new_col(j, :) = string(col_temp{j, idx_col});
        end
    end

    %Modify table
    table_main2(:, {item, item2}) = [];
    table_main2 = addvars(table_main2, new_col, ...
        'NewVariableNames', item, 'Before',item_idx);
end


%% Homogenise names (mismatch between tables)

table_external2 = renamevars(table_external, ...
    {'Zone', 'Age_Ma', 'Age_2SE_Ma', 'Deposit'}, ...
    {'Position', 'Zr_Age_Ma', 'Zr_Age_2SE_Ma', 'Deposit/Batholith'});

varNames1 = table_main2.Properties.VariableNames;
varNames2 = table_external2.Properties.VariableNames;

%medicine 1: all to strings
table_main3 = convertvars(table_main2, varNames1,'string');
table_external3 = convertvars(table_external2, varNames2,'string');

%medicine 2: replace missing by NA
idx_main = ismissing(table_main3);
str1 = table2array(table_main3);
str1(idx_main) = 'NA';
table_main4 = array2table(str1, "VariableNames", table_main3.Properties.VariableNames);

idx_external = ismissing(table_external3);
str2 = table2array(table_external3);
str2(idx_external) = 'NA';
table_external4 = array2table(str2, "VariableNames", table_external3.Properties.VariableNames);

%% Merging tables

%Set identifier
table_main4.setID = string(repmat('main', [size(table_main4, 1), 1]));
table_external4.setID = string(repmat('validation', [size(table_external4, 1), 1]));

table_merged0 = tblvertcat(table_main4, table_external4);

%medicine 3: replace missing by NA
idx_main = ismissing(table_merged0);
str3 = table2array(table_merged0);
str3(idx_main) = 'NA';
table_merged = array2table(str3, "VariableNames", table_merged0.Properties.VariableNames);

%% Re-arranging


mingled_columns = {'ml_classes', 'setID', 'Zircon.Name'};
table_merged1 = movevars(table_merged, mingled_columns, 'Before', 1);

numeric_idx = 21;%feedback from seeing Workspace variable
Vars1 = table_merged1.Properties.VariableNames;
Vars2 = Vars1(numeric_idx:end);

%appending numeric

mtx = double(table_merged1{:, numeric_idx:end});
append_table = array2table(mtx, 'VariableNames',Vars2);

table_merged2 = table_merged1(:, 1:numeric_idx-1);
table_merged3 = [table_merged2, append_table];

%% Further homogenisation (for future reference)

%Spot analysis position
position_temp = table_merged3.Position;

str_core = ["CORE", "Core", "c", "core", "interior"];
str_middle = ["MID", "Middle", "m"];
str_rim = ["RIM", "Rim", "r", "rim", "rim1", "rim2", "rim3", "exterior"];
str_mixed = ["d.z.", "interior/r", "rim/core"];

[idx1, ~] = ismember(position_temp, str_core);
[idx2, ~] = ismember(position_temp, str_middle);
[idx3, ~] = ismember(position_temp, str_rim);
[idx4, ~] = ismember(position_temp, str_mixed);

position_temp(idx1) = "core";
position_temp(idx2) = "mantle";
position_temp(idx3) = "rim";
position_temp(idx4) = "mixed";

table_merged3.Position = position_temp;

%Temporality of mineralization
position_temp = table_merged3.Temporality;

str_syn = ["Syn Mineralisation"];
str_pre = ["Pre Mineralisation"];

[idx1, ~] = ismember(position_temp, str_syn);
[idx2, ~] = ismember(position_temp, str_pre);

position_temp(idx1) = "Syn Mineral";
position_temp(idx2) = "Pre Mineral";

table_merged3.Temporality = position_temp;

%Composition
position_temp = table_merged3.Composition;
str_1 = ["NA"];
[idx1, ~] = ismember(position_temp, str_1);
idx2 = ismissing(position_temp);
idx = idx1 | idx2;
position_temp(idx) = "Unknown";
table_merged3.Composition = position_temp;

%Continent
position_temp = table_merged3.Continent;
str_1 = ["NA"];
[idx1, ~] = ismember(position_temp, str_1);
idx2 = ismissing(position_temp);
idx = idx1 | idx2;
position_temp(idx) = "Unknown";
table_merged3.Continent = position_temp;

%District
position_temp = table_merged3.District;
str_1 = ["NA"];
[idx1, ~] = ismember(position_temp, str_1);
idx2 = ismissing(position_temp);
idx = idx1 | idx2;
position_temp(idx) = "Unknown";
table_merged3.District = position_temp;

%Dataset
position_temp = table_merged3.Dataset;
str_1 = ["NA"];
[idx1, ~] = ismember(position_temp, str_1);
idx2 = ismissing(position_temp);
idx = idx1 | idx2;
position_temp(idx) = "Unknown";
table_merged3.Dataset = position_temp;

% %Optional: Timing (tries to group Temporality)
% position_temp = table_merged3.Temporality;
% 
% str_1 = ["Post Mineral", "Pre Mineral", "Precursor", "Pre to Syn Mineral", "Syn to Post Mineral", "Ore Related Magmatism"];
% str_2 = ["NA"];
% str_3 = ["Syn Mineral"];
% 
% [idx1, ~] = ismember(position_temp, str_1);
% [idx2_1, ~] = ismember(position_temp, str_2);
% idx2_2 = ismissing(position_temp);
% [idx3, ~] = ismember(position_temp, str_3);
% 
% position_temp(idx1) = "Ore related magmatism";
% position_temp(idx2_1 | idx2_2) = "Unknown";
% position_temp(idx3) = "Ore syn-mineral magmatism";
% table_merged3.Timing = position_temp;
% % unique(position_temp)

%%
%saving
file3 = 'CG2024data_v5.csv';
filepath3 = fullfile(workingDir, file3);
writetable(table_merged3, filepath3)


