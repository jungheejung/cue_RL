%% Model Fitting, Cognitive:
% AUTHOR: ARYAN YAZDANPANAH
% For each subject:
% 
% 1)	Calculate the Noxious inputs (1, 2, or 3) for each of the participants based on the average ratings of that participant on each of those real Noxious inputs. For example, to calculate the subjective Noxious 1 (48) for participant 1, we average the ratings on the trials that the participant has got 48-degree heat (and the pain rating is reported on 0-180 scale). 
% From pain rating vector: 
% for each Noxious =(48, 49, or 50):
% 	N (1, 2, or 3) = mean (pain_rating (Noxious))
% End
% 2)	Calculate the Expectation and Pain matrices based on each model
% 3)	Calculate the NLL based on the above equation
% 4)	For jj=1:20 % run 20 times to avoid being stuck in the 
% Optimize the NLL function and get the parameters and NLL for each model and pick the best parameter
% End
% 5)	Calculate aic, bic, and NLL or each model and compare different models together
%% do the model fitting part (nll=nll_Pain+nll_Exp)
clear
close all
clc

run_numbers=20;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Excluding particpants with few trials or with less than 3 sessions:
PATH_long=[cd '\long_frmt_R'];
table_pain=readtable([PATH_long '\spacetop_cue_cognitive_0405.csv']);
subj_num=unique(table_pain.src_subject_id);
for ii=1:length(subj_num)
    session_id{ii}=unique(table_pain.session_id(table_pain.src_subject_id==subj_num(ii)));
    num_trial_subj(ii)=max(table_pain.trial_index_subjectwise(table_pain.src_subject_id==subj_num(ii)));
    num_session_subj(ii)=length(session_id{ii});
end
% min(num_trial_subj)
% max(num_trial_subj)
subj_num_new=subj_num;
subj_num_new(num_trial_subj<48 | num_session_subj<3)=[]; % excluding the subjects that have less than 48 trials or less than 3 sessions
selected_rows = ismember(table_pain.src_subject_id, subj_num_new);
table_pain_new=table_pain(selected_rows,:); % creating the new table that exclude subjects with less than 50 trials

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Now I fit the model:
% defining initial valus for the parameters:
% mdl1) (2 parameters) 'mdl1': (1) one alpha (2) one constant weight: w 
% (3) painerror (4) experror 
lb{1}=[0 0 0 0];
ub{1}=[1 1 100 100];
% mdl2) (3 parameters) 'mdl2 (confirmation bias with one weight)': (1) alpha_c
% (2) alpha_i (3) constant weight: w (4) painerror (5) experror
lb{2}=[0 0 0 0 0];
ub{2}=[1 1 1 100 100];
% mdl3)(2 parameters) 'mdl3 (one alpha with changing weight)':  (1) one alpha
% (2) one changing weight: w=2/(1+exp(gamma*|N-E|)). So parameters are
% alpha and gamma and (3) painerror (4) experror:
lb{3}=[0 0 0 0];
ub{3}=[1 100 100 100];
% mdl4)(3 parameters) 'mdl4 (confirmation bias with one changing weight)': (1)
% (1) alpha_c (2) alpha_i (3) one changing weight: w=2/(1+exp(gamma*|N-E|)). So parameters are
% alpha_c and alpha_i and gamma and % (4) painerror (5) experror
lb{4}=[0 0 0 0 0];
ub{4}=[1 1 100 100 100];
model_type={'mdl1', 'mdl2', 'mdl3', 'mdl4'};


% for each subject:
qpar=cell(length(subj_num_new),length(model_type),run_numbers);
for ii=1:length(subj_num_new)
ii
sub_no=subj_num_new(ii);
% 0) get the data matrix for the subject:
Data=table_pain_new(table_pain_new.src_subject_id==sub_no,:);
Data_lowpain=Data(strcmp(Data.param_stimulus_type, 'low_stim'),:);
Data_medpain=Data(strcmp(Data.param_stimulus_type, 'med_stim'),:);
Data_highpain=Data(strcmp(Data.param_stimulus_type, 'high_stim'),:);
% 1)	Calculate the Noxious inputs (1, 2, or 3) for each of the participants based on the average ratings of that participant on each of those real Noxious inputs. For example, to calculate the subjective Noxious 1 (48) for participant 1, we average the ratings on the trials that the participant has got 48-degree heat (and the pain rating is reported on 0-180 scale). 
% From pain rating vector: 
N(ii,1)=nanmean(Data_lowpain.event04_actual_angle);
N(ii,2)=nanmean(Data_medpain.event04_actual_angle);
N(ii,3)=nanmean(Data_highpain.event04_actual_angle);
% Noxious input vecotr for each participant:
Noxious{ii} = zeros(length(Data.param_stimulus_type),1);
Noxious{ii}(strcmp(Data.param_stimulus_type,'low_stim')) = N(ii,1);
Noxious{ii}(strcmp(Data.param_stimulus_type,'med_stim')) = N(ii,2);
Noxious{ii}(strcmp(Data.param_stimulus_type,'high_stim')) = N(ii,3);
% High_Cue vector for each participant:
High_Cue{ii} = strcmp(Data.param_cue_type,'high_cue');
% 2, 3, 4)	Calculate the Expectation and Pain matrices based on each model
% Calculate the nll based on the above equation
% For jj=1:20 % run 20 times to avoid being stuck in the 
%   Optimize the nll function and get the parameters and nll for each model and pick the best parameter
% End
% 5)	Calculate aic, bic, and nll or each model and compare different models together
Noxious_subj=Noxious{ii};
High_Cue_subj=High_Cue{ii};

        m=1;
        for model_type={'mdl1', 'mdl2', 'mdl3', 'mdl4'}
            fit_fun=@(xpar)NLL(xpar, Data, model_type{1},Noxious_subj,High_Cue_subj);
            for jj=1:run_numbers
                initpar=lb{m}+rand(1,length(lb{m})).*(ub{m}-lb{m});
                [qpar{ii,m,jj}, nll(ii,m,jj), bic(ii,m,jj), nlike, aic(ii,m,jj)]=fit_fun_2(Data,fit_fun,initpar,lb{m},ub{m});
            end
            m=m+1; 
        end
end

save cognitive_model_fitting_NLL

%% model comparisons:
for ii=1:size(aic,1)
    idx_best=find(aic(ii,1,:)==min(aic(ii,1,:)));
    aic_mat(ii,1)=aic(ii,1,idx_best(1));
        idx_best=find(aic(ii,2,:)==min(aic(ii,2,:)));
    aic_mat(ii,2)=aic(ii,2,idx_best(1));
        idx_best=find(aic(ii,3,:)==min(aic(ii,3,:)));
    aic_mat(ii,3)=aic(ii,3,idx_best(1));
        idx_best=find(aic(ii,4,:)==min(aic(ii,4,:)));
    aic_mat(ii,4)=aic(ii,4,idx_best(1));
end

for ii=1:size(nll,1)
    idx_best=find(nll(ii,1,:)==min(nll(ii,1,:)));
    nll_mat(ii,1)=nll(ii,1,idx_best(1));
        idx_best=find(nll(ii,2,:)==min(nll(ii,2,:)));
    nll_mat(ii,2)=nll(ii,2,idx_best(1));
        idx_best=find(nll(ii,3,:)==min(nll(ii,3,:)));
    nll_mat(ii,3)=nll(ii,3,idx_best(1));
        idx_best=find(nll(ii,4,:)==min(nll(ii,4,:)));
    nll_mat(ii,4)=nll(ii,4,idx_best(1));
end

% compare model 1 and 3 in terms of aic and nll:
% aic:
figure
histogram(aic_mat(:,2)-aic_mat(:,4),20)
title('aic difference between model 2 and model 4')
% nll:
figure
histogram(nll_mat(:,2)-nll_mat(:,4),20)
title('nll difference between model 2 and model 4')

% comparing model 1 and 2 in terms of aic and nll
% aic:
figure
histogram(aic_mat(:,1)-aic_mat(:,2),200)
title('aic difference between model 1 and model 2')
% nll:
figure
histogram(nll_mat(:,1)-nll_mat(:,2),200)
title('nll difference between model 1 and model 2')




figure
column_names = {'mdl1: w constant & no biased learning', ...
                'mdl2: w constant & biased learning', ...
                'mdl3: w change & no biased learning', ...
                'mdl4: w change & biased learning'};
bar(mean(nll_mat))
xticklabels(column_names);
xtickangle(45);
ylim([527,536])
title('mean of nll in each model')


figure
aic_mat_mean_cen=aic_mat-mean(aic_mat,2);
within_error_aic=std(aic_mat_mean_cen,0,1)/(size(aic_mat_mean_cen,1)-2);
column_names = {'mdl1: w constant & no biased learning', ...
                'mdl2: w constant & biased learning', ...
                'mdl3: w change & no biased learning', ...
                'mdl4: w change & biased learning'};
bar(mean(aic_mat))
hold on
errorbar([1,2,3,4],mean(aic_mat),within_error_aic,'.r')
xticklabels(column_names);
xtickangle(45);
ylim([1060,1080])
title('mean of aic in each model')
p12=signrank(aic_mat(:,1),aic_mat(:,2));
p13=signrank(aic_mat(:,1),aic_mat(:,3));
p14=signrank(aic_mat(:,1),aic_mat(:,4));
p23=signrank(aic_mat(:,2),aic_mat(:,3));
p24=signrank(aic_mat(:,2),aic_mat(:,4));
p34=signrank(aic_mat(:,3),aic_mat(:,4));
sigstar({[1, 2], [1, 3], [1, 4], [2,3], [2, 4],[3, 4]}, [p12,p13,p14,p23,p24,p34]);
%% looking at the the model 4:
for ii=1:size(aic,1)
    idx_best=find(aic(ii,4,:)==min(aic(ii,4,:)));
    param_mdl4(:,ii)=qpar{ii,4,idx_best(1)};
end

figure
scatter(param_mdl4(1,:),param_mdl4(2,:),'filled')
xlim([0,1])
ylim([0,1])
title('learning rates')
xlabel('alpha congruent')
ylabel('alpha incongruent')


% simulating the 63 subjects:
% defining parameters:
for jj=1:size(param_mdl4,2)
    Data=table_pain_new(table_pain_new.src_subject_id==subj_num_new(jj),:);
    idx_ExpH = find(strcmp(Data.param_cue_type, 'high_cue'), 1);
idx_ExpL = find(strcmp(Data.param_cue_type, 'low_cue'), 1);
first_exp_high_cue(jj) = Data.event02_expect_angle(idx_ExpH);
first_exp_low_cue(jj) = Data.event02_expect_angle(idx_ExpL);
        alpha_c=param_mdl4(1,jj);
        alpha_i=param_mdl4(2,jj);
        gamma=param_mdl4(3,jj);
        % defining expectations and pains:
        num_trial=length(Noxious{jj});
        ExpectationH{jj}=first_exp_high_cue(jj)*ones(num_trial+1,1); % Expectation_highCue matrix
        ExpectationL{jj}=first_exp_low_cue(jj)*ones(num_trial+1,1); % Expectation_lowCue matrix
        Pain{jj}=zeros(num_trial+1,1); % Pain matrix
        PE{jj}=zeros(num_trial+1,1); % Pain predcition error matrix
        PainH{jj}=zeros(num_trial+1,1); % Pain matrix for high cues
        PainL{jj}=zeros(num_trial+1,1); % Pain matrix for low cues
        for ii=1:num_trial
            if High_Cue{jj}(ii)==1
                w=2/(1+exp(gamma*abs(Noxious{jj}(ii,1)-ExpectationH{jj}(ii,1))));
                Pain{jj}(ii,1)=(1-w)*Noxious{jj}(ii,1)+w*ExpectationH{jj}(ii,1);
                PE{jj}(ii,1)=Pain{jj}(ii,1)-ExpectationH{jj}(ii,1);
                if PE{jj}(ii,1)>=0
                    alpha=alpha_c;
                else
                    alpha=alpha_i;
                end
                ExpectationH{jj}(ii+1,1)=ExpectationH{jj}(ii,1)+alpha*PE{jj}(ii,1);
                ExpectationL{jj}(ii+1,1)=ExpectationL{jj}(ii,1);
                PainH{jj}(ii,1)=Pain{jj}(ii,1);
                PainL{jj}(ii,1)=nan;
                nlls{jj}(ii) = (Pain{jj}(ii,1)-Data.event04_actual_angle(ii))^2+(ExpectationH{jj}(ii,1)-Data.event02_expect_angle(ii))^2;
            else
                w=2/(1+exp(gamma*abs(Noxious{jj}(ii,1)-ExpectationL{jj}(ii,1))));
                Pain{jj}(ii,1)=(1-w)*Noxious{jj}(ii,1)+w*ExpectationL{jj}(ii,1);
                PE{jj}(ii,1)=Pain{jj}(ii,1)-ExpectationL{jj}(ii,1);
                if PE{jj}(ii,1)>=0
                    alpha=alpha_i;
                else
                    alpha=alpha_c;
                end
                ExpectationL{jj}(ii+1,1)=ExpectationL{jj}(ii,1)+alpha*PE{jj}(ii,1);
                ExpectationH{jj}(ii+1,1)=ExpectationH{jj}(ii,1);
                PainL{jj}(ii,1)=Pain{jj}(ii,1);
                PainH{jj}(ii,1)=nan;
                nlls{jj}(ii) = (Pain{jj}(ii,1)-Data.event04_actual_angle(ii))^2+(ExpectationL{jj}(ii,1)-Data.event02_expect_angle(ii))^2;
            end
            
        end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first, I add columns tothe table_pain_new that are based on the pain and
% expectation levels of the models:
Pain_mdl4 = [];
for i = 1:numel(Pain)
    vector = Pain{i};
    vector(end) = []; % remove last element
    mean_Pain(i)=nanmean(vector);
    Pain_mdl4 = [Pain_mdl4;vector];
end
table_pain_new.Pain_mdl4=Pain_mdl4;

PainH_mdl4 = [];
for i = 1:numel(PainH)
    vector = PainH{i};
    vector(end) = []; % remove last element
    mean_PainH(i)=nanmean(vector);
    PainH_mdl4 = [PainH_mdl4;vector];
end
table_pain_new.PainH_mdl4=PainH_mdl4;

PainL_mdl4 = [];
for i = 1:numel(PainL)
    vector = PainL{i};
    vector(end) = []; % remove last element
    mean_PainL(i)=nanmean(vector);
    PainL_mdl4 = [PainL_mdl4;vector];
end
table_pain_new.PainL_mdl4=PainL_mdl4;

expH_mdl4 = [];
for i = 1:numel(ExpectationH)
    vector = ExpectationH{i};
    vector(end) = []; % remove last element
    mean_ExpectationH(i)=nanmean(vector);
    expH_mdl4 = [expH_mdl4;vector];
end
table_pain_new.expH_mdl4=expH_mdl4;

expL_mdl4 = [];
for i = 1:numel(ExpectationL)
    vector = ExpectationL{i};
    vector(end) = []; % remove last element
    mean_ExpectationL(i)=nanmean(vector);
    expL_mdl4 = [expL_mdl4;vector];
end
table_pain_new.expL_mdl4=expL_mdl4;


table_pain_new.exp_mdl4 = NaN(size(table_pain_new,1),1);
table_pain_new.exp_mdl4(strcmp(table_pain_new.param_cue_type,'high_cue')) = table_pain_new.expH_mdl4(strcmp(table_pain_new.param_cue_type,'high_cue'));
table_pain_new.exp_mdl4(strcmp(table_pain_new.param_cue_type,'low_cue')) = table_pain_new.expL_mdl4(strcmp(table_pain_new.param_cue_type,'low_cue'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% % subject 1:
% figure
% kk=1;
% plot(ExpectationH{kk}(1:num_trial,:),'--r')
% hold on
% plot(PainH{kk}(1:num_trial,:),'r')
% hold on
% plot(PainL{kk}(1:num_trial,:),'b')
% hold on
% plot(ExpectationL{kk}(1:num_trial,:),'--b')
% legend('ExpH','PainH','PainL','ExpL')


% mean of the expectations and pain reports for each model (subject)
figure
scatter(mean_ExpectationH,mean_PainH,'filled','r')
hold on
scatter(mean_ExpectationL,mean_PainL,'filled','b')
hold on
plot([0, 180], [0, 180])
xlim([0,180])
ylim([0,180])
legend({'High Cue', 'Low Cue'}) 
title('pain rating and expectation ratings in high and low cue')
xlabel('Mean Expectation')
ylabel('Mean Pain Outcome')


figure
histogram(mean_PainH-mean_PainL,20)
xline(median(mean_PainH-mean_PainL),'r')
xline(mean(mean_PainH-mean_PainL),'--r')
title('histogram of difference between PainH and PainL')
[h,p]=ttest(mean_PainH-mean_PainL);
p_med=signrank(mean_PainH-mean_PainL);
subtitle(['mean = ' num2str(mean(mean_PainH-mean_PainL)) ' & pvalue = ' num2str(p) 'median = ' num2str(median(mean_PainH-mean_PainL)) ' & pvalue = ' num2str(p_med)])


figure
histogram(mean_ExpectationH-mean_ExpectationL,20)
xline(median(mean_ExpectationH-mean_ExpectationL),'r')
xline(mean(mean_ExpectationH-mean_ExpectationL),'--r')
title('histogram of difference between ExpectationH and ExpectationL')
[h,p]=ttest(mean_ExpectationH-mean_ExpectationL);
p_med=signrank(mean_ExpectationH-mean_ExpectationL);
subtitle(['mean = ' num2str(mean(mean_ExpectationH-mean_ExpectationL)) ' & pvalue = ' num2str(p) 'median = ' num2str(median(mean_ExpectationH-mean_ExpectationL)) ' & pvalue = ' num2str(p_med)])




% Doing the analysis in three different stimulus types:
%%%%% NOW looking at the behaiovr of the model in three different stimulus
%%%%% intensities:
% in three different stimulus intensitites:
% create a figure with three subplots
stim_type={'low_stim','med_stim','high_stim'};
figure;
for i = 1:3
    % filter the data to include only rows with the current stimulus type
    subset = table_pain_new(strcmp(table_pain_new.param_stimulus_type, stim_type{i}), :);
    % calculate the averages for each ID in each high/low cue
    averages = grpstats(subset, {'src_subject_id', 'param_cue_type'}, {'mean'}, 'DataVars', {'Pain_mdl4', 'exp_mdl4'});
    % create a scatter plot with different colors for high and low cue
    for ii = 1:size(averages, 1)
    if strcmp(averages.param_cue_type{ii}, 'high_cue')
        colors(ii,:) = [1 0 0]; % set red color for high cue
    else
       colors(ii,:) = [0 0 1]; % blue
    end
    end
    subplot(1, 3, i);
    scatter(averages.mean_exp_mdl4,averages.mean_Pain_mdl4, [], colors, 'filled');
    title(['Stimulus Type ', num2str(i)]);
    xlabel('Mean Pain of model');
    ylabel('Mean Expectation of model');
    
end



% On demean data:
% in three different stimulus intensitites:
% create a figure with three subplots
stim_type={'low_stim','med_stim','high_stim'};
meanPain_subj_mdl4 = groupsummary(table_pain_new,'src_subject_id','mean','Pain_mdl4');
meanexp_subj_mdl4 = groupsummary(table_pain_new,'src_subject_id','mean','exp_mdl4');
table_pain_new.paindmean=zeros(size(table_pain_new,1),1);
table_pain_new.expdmean=zeros(size(table_pain_new,1),1);
for ii=1:size(subj_num_new,1)
    idx_subj=table_pain_new.src_subject_id==subj_num_new(ii);
table_pain_new.paindmean_mdl4(idx_subj)=table_pain_new.Pain_mdl4(idx_subj)-meanPain_subj_mdl4.mean_Pain_mdl4(ii);
table_pain_new.expectdmean_mdl4(idx_subj)=table_pain_new.exp_mdl4(idx_subj)-meanexp_subj_mdl4.mean_exp_mdl4(ii);
end

figure;
for i = 1:3
    % filter the data to include only rows with the current stimulus type
    subset = table_pain_new(strcmp(table_pain_new.param_stimulus_type, stim_type{i}), :);
    % calculate the averages for each ID in each high/low cue
    averages = grpstats(subset, {'src_subject_id', 'param_cue_type'}, {'mean'}, 'DataVars', {'expectdmean_mdl4', 'paindmean_mdl4'});
    % create a scatter plot with different colors for high and low cue
    for ii = 1:size(averages, 1)
    if strcmp(averages.param_cue_type{ii}, 'high_cue')
        colors(ii,:) = [1 0 0]; % set red color for high cue
    else
       colors(ii,:) = [0 0 1]; % blue
    end
    end
    subplot(1, 3, i);
    scatter(averages.mean_expectdmean_mdl4,averages.mean_paindmean_mdl4, [], colors, 'filled');
    title(['Stimulus Type ', num2str(i)]);
    xlabel('demean Pain model 4');
    ylabel('demean Expectation model 4');
    ylim([-60,50])
    xlim([-60,50])
    
end


figure;
for i = 1:3
    % filter the data to include only rows with the current stimulus type
    subset = table_pain_new(strcmp(table_pain_new.param_stimulus_type, stim_type{i}), :);
    % calculate the averages for each ID in each high/low cue
    averages = grpstats(subset, {'src_subject_id', 'param_cue_type'}, {'mean'}, 'DataVars', {'expectdmean_mdl4', 'paindmean_mdl4'});
    dmeanpain_mdl4(:,i,1)=averages.mean_paindmean_mdl4(strcmp(averages.param_cue_type, 'low_cue')); % low cue
    dmeanpain_mdl4(:,i,2)=averages.mean_paindmean_mdl4(strcmp(averages.param_cue_type, 'high_cue')); % high cue
    dmeanexp_mdl4(:,i,1)=averages.mean_expectdmean_mdl4(strcmp(averages.param_cue_type, 'low_cue')); % low cue
    dmeanexp_mdl4(:,i,2)=averages.mean_expectdmean_mdl4(strcmp(averages.param_cue_type, 'high_cue')); % high cue
    
    % create a scatter plot with different colors for high and low cue
    for ii = 1:size(averages, 1)
    if strcmp(averages.param_cue_type{ii}, 'high_cue')
        colors(ii,:) = [1 0 0]; % set red color for high cue
    else
       colors(ii,:) = [0 0 1]; % blue
    end
    end
    subplot(1, 3, i);
    scatter(averages.mean_expectdmean_mdl4,averages.mean_paindmean_mdl4, [], colors, 'filled');
    title(['Stimulus Type ', num2str(i)]);
    [r_ILC,p_ILC]=corr(dmeanexp_mdl4(:,i,1),dmeanpain_mdl4(:,i,1),'type','Spearman');
[r_IHC,p_IHC]=corr(dmeanexp_mdl4(:,i,2),dmeanpain_mdl4(:,i,2),'type','Spearman');
subtitle_text = sprintf('corrL=%d & corrH=%d\npvalL=%d & pvalH=%d',r_ILC ,r_IHC ,p_ILC ,p_IHC);
subtitle(subtitle_text)    
    xlabel('demean Expectation model 4');
    ylabel('demean Pain model 4');
    ylim([-60,50])
    xlim([-60,50])
    
end



%% looking at the the model 1:
for ii=1:size(aic,1)
    idx_best=find(aic(ii,1,:)==min(aic(ii,1,:)));
    param_mdl1(:,ii)=qpar{ii,1,idx_best(1)};
end

figure
scatter(param_mdl1(1,:),param_mdl1(2,:),'filled')
xlim([0,1])
ylim([0,1])
title('learning rate and W_E')
xlabel('alpha')
ylabel('weight of Expectation')


% simulating the 63 subjects:
% defining parameters:
for jj=1:size(param_mdl1,2)
    Data=table_pain_new(table_pain_new.src_subject_id==subj_num_new(jj),:);
       idx_ExpH = find(strcmp(Data.param_cue_type, 'high_cue'), 1);
idx_ExpL = find(strcmp(Data.param_cue_type, 'low_cue'), 1);
first_exp_high_cue(jj) = Data.event02_expect_angle(idx_ExpH);
first_exp_low_cue(jj) = Data.event02_expect_angle(idx_ExpL);
        alpha=param_mdl1(1,jj);
        w=param_mdl1(2,jj);

        % defining expectations and pains:
        num_trial=length(Noxious{jj});
        ExpectationH{jj}=first_exp_high_cue(jj)*ones(num_trial+1,1); % Expectation_highCue matrix
        ExpectationL{jj}=first_exp_low_cue(jj)*ones(num_trial+1,1); % Expectation_lowCue matrix
        Pain{jj}=zeros(num_trial+1,1); % Pain matrix
        PE{jj}=zeros(num_trial+1,1); % Pain predcition error matrix
        PainH{jj}=zeros(num_trial+1,1); % Pain matrix for high cues
        PainL{jj}=zeros(num_trial+1,1); % Pain matrix for low cues
        for ii=1:num_trial
            if High_Cue{jj}(ii)==1
                Pain{jj}(ii,1)=(1-w)*Noxious{jj}(ii,1)+w*ExpectationH{jj}(ii,1);
                PE{jj}(ii,1)=Pain{jj}(ii,1)-ExpectationH{jj}(ii,1);
                ExpectationH{jj}(ii+1,1)=ExpectationH{jj}(ii,1)+alpha*PE{jj}(ii,1);
                ExpectationL{jj}(ii+1,1)=ExpectationL{jj}(ii,1);
                PainH{jj}(ii,1)=Pain{jj}(ii,1);
                PainL{jj}(ii,1)=nan;
                nlls{jj}(ii) = (Pain{jj}(ii,1)-Data.event04_actual_angle(ii))^2+(ExpectationH{jj}(ii,1)-Data.event02_expect_angle(ii))^2;
            else
                Pain{jj}(ii,1)=(1-w)*Noxious{jj}(ii,1)+w*ExpectationL{jj}(ii,1);
                PE{jj}(ii,1)=Pain{jj}(ii,1)-ExpectationL{jj}(ii,1);
                ExpectationL{jj}(ii+1,1)=ExpectationL{jj}(ii,1)+alpha*PE{jj}(ii,1);
                ExpectationH{jj}(ii+1,1)=ExpectationH{jj}(ii,1);
                PainL{jj}(ii,1)=Pain{jj}(ii,1);
                PainH{jj}(ii,1)=nan;
                nlls{jj}(ii) = (Pain{jj}(ii,1)-Data.event04_actual_angle(ii))^2+(ExpectationL{jj}(ii,1)-Data.event02_expect_angle(ii))^2;
            end
            
        end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first, I add columns tothe table_pain_new that are based on the pain and
% expectation levels of the models:
Pain_mdl1 = [];
for i = 1:numel(Pain)
    vector = Pain{i};
    vector(end) = []; % remove last element
    mean_Pain(i)=nanmean(vector);
    Pain_mdl1 = [Pain_mdl1;vector];
end
table_pain_new.Pain_mdl1=Pain_mdl1;

PainH_mdl1 = [];
for i = 1:numel(PainH)
    vector = PainH{i};
    vector(end) = []; % remove last element
    mean_PainH(i)=nanmean(vector);
    PainH_mdl1 = [PainH_mdl1;vector];
end
table_pain_new.PainH_mdl1=PainH_mdl1;

PainL_mdl1 = [];
for i = 1:numel(PainL)
    vector = PainL{i};
    vector(end) = []; % remove last element
    mean_PainL(i)=nanmean(vector);
    PainL_mdl1 = [PainL_mdl1;vector];
end
table_pain_new.PainL_mdl1=PainL_mdl1;

expH_mdl1 = [];
for i = 1:numel(ExpectationH)
    vector = ExpectationH{i};
    vector(end) = []; % remove last element
    mean_ExpectationH(i)=nanmean(vector);
    expH_mdl1 = [expH_mdl1;vector];
end
table_pain_new.expH_mdl1=expH_mdl1;

expL_mdl1 = [];
for i = 1:numel(ExpectationL)
    vector = ExpectationL{i};
    vector(end) = []; % remove last element
    mean_ExpectationL(i)=nanmean(vector);
    expL_mdl1 = [expL_mdl1;vector];
end
table_pain_new.expL_mdl1=expL_mdl1;


table_pain_new.exp_mdl1 = NaN(size(table_pain_new,1),1);
table_pain_new.exp_mdl1(strcmp(table_pain_new.param_cue_type,'high_cue')) = table_pain_new.expH_mdl1(strcmp(table_pain_new.param_cue_type,'high_cue'));
table_pain_new.exp_mdl1(strcmp(table_pain_new.param_cue_type,'low_cue')) = table_pain_new.expL_mdl1(strcmp(table_pain_new.param_cue_type,'low_cue'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % subject 1:
% figure
% kk=1;
% plot(ExpectationH{kk}(1:num_trial,:),'--r')
% hold on
% plot(PainH{kk}(1:num_trial,:),'r')
% hold on
% plot(PainL{kk}(1:num_trial,:),'b')
% hold on
% plot(ExpectationL{kk}(1:num_trial,:),'--b')
% legend('ExpH','PainH','PainL','ExpL')


% mean of the expectations and pain reports for each model (subject)
figure
scatter(mean_ExpectationH,mean_PainH,'filled','r')
hold on
scatter(mean_ExpectationL,mean_PainL,'filled','b')
hold on
plot([0, 180], [0, 180])
xlim([0,180])
ylim([0,180])
legend({'High Cue', 'Low Cue'}) 
title('pain rating and expectation ratings in high and low cue')
xlabel('Mean Expectation')
ylabel('Mean Pain Outcome')


figure
histogram(mean_PainH-mean_PainL,20)
xline(median(mean_PainH-mean_PainL),'r')
xline(mean(mean_PainH-mean_PainL),'--r')
title('histogram of difference between PainH and PainL')
[h,p]=ttest(mean_PainH-mean_PainL);
p_med=signrank(mean_PainH-mean_PainL);
subtitle(['mean = ' num2str(mean(mean_PainH-mean_PainL)) ' & pvalue = ' num2str(p) 'median = ' num2str(median(mean_PainH-mean_PainL)) ' & pvalue = ' num2str(p_med)])


figure
histogram(mean_ExpectationH-mean_ExpectationL,20)
xline(median(mean_ExpectationH-mean_ExpectationL),'r')
xline(mean(mean_ExpectationH-mean_ExpectationL),'--r')
title('histogram of difference between ExpectationH and ExpectationL')
[h,p]=ttest(mean_ExpectationH-mean_ExpectationL);
p_med=signrank(mean_ExpectationH-mean_ExpectationL);
subtitle(['mean = ' num2str(mean(mean_ExpectationH-mean_ExpectationL)) ' & pvalue = ' num2str(p) 'median = ' num2str(median(mean_ExpectationH-mean_ExpectationL)) ' & pvalue = ' num2str(p_med)])


%%%%% NOW looking at the behaiovr of the model in three different stimulus
%%%%% intensities:
% in three different stimulus intensitites:
% create a figure with three subplots
stim_type={'low_stim','med_stim','high_stim'};
figure;
for i = 1:3
    % filter the data to include only rows with the current stimulus type
    subset = table_pain_new(strcmp(table_pain_new.param_stimulus_type, stim_type{i}), :);
    % calculate the averages for each ID in each high/low cue
    averages = grpstats(subset, {'src_subject_id', 'param_cue_type'}, {'mean'}, 'DataVars', {'Pain_mdl1', 'exp_mdl1'});
    % create a scatter plot with different colors for high and low cue
    for ii = 1:size(averages, 1)
    if strcmp(averages.param_cue_type{ii}, 'high_cue')
        colors(ii,:) = [1 0 0]; % set red color for high cue
    else
       colors(ii,:) = [0 0 1]; % blue
    end
    end
    subplot(1, 3, i);
    scatter(averages.mean_exp_mdl1,averages.mean_Pain_mdl1, [], colors, 'filled');
    title(['Stimulus Type ', num2str(i)]);
    xlabel('Mean Pain of model');
    ylabel('Mean Expectation of model');
    
end




% On demean data:
% in three different stimulus intensitites:
% create a figure with three subplots
stim_type={'low_stim','med_stim','high_stim'};
meanPain_subj_mdl1 = groupsummary(table_pain_new,'src_subject_id','mean','Pain_mdl1');
meanexp_subj_mdl1 = groupsummary(table_pain_new,'src_subject_id','mean','exp_mdl1');
table_pain_new.paindmean=zeros(size(table_pain_new,1),1);
table_pain_new.expdmean=zeros(size(table_pain_new,1),1);
for ii=1:size(subj_num_new,1)
    idx_subj=table_pain_new.src_subject_id==subj_num_new(ii);
table_pain_new.paindmean_mdl1(idx_subj)=table_pain_new.Pain_mdl1(idx_subj)-meanPain_subj_mdl1.mean_Pain_mdl1(ii);
table_pain_new.expectdmean_mdl1(idx_subj)=table_pain_new.exp_mdl1(idx_subj)-meanexp_subj_mdl1.mean_exp_mdl1(ii);
end

figure;
for i = 1:3
    % filter the data to include only rows with the current stimulus type
    subset = table_pain_new(strcmp(table_pain_new.param_stimulus_type, stim_type{i}), :);
    % calculate the averages for each ID in each high/low cue
    averages = grpstats(subset, {'src_subject_id', 'param_cue_type'}, {'mean'}, 'DataVars', {'expectdmean_mdl1', 'paindmean_mdl1'});
    % create a scatter plot with different colors for high and low cue
    for ii = 1:size(averages, 1)
    if strcmp(averages.param_cue_type{ii}, 'high_cue')
        colors(ii,:) = [1 0 0]; % set red color for high cue
    else
       colors(ii,:) = [0 0 1]; % blue
    end
    end
    subplot(1, 3, i);
    scatter(averages.mean_expectdmean_mdl1,averages.mean_paindmean_mdl1, [], colors, 'filled');
    title(['Stimulus Type ', num2str(i)]);
    xlabel('demean Pain model 1');
    ylabel('demean Expectation model 1');
    ylim([-60,50])
    xlim([-60,50])
    
end


%% looking at the the model 2:
for ii=1:size(aic,1)
    idx_best=find(aic(ii,2,:)==min(aic(ii,2,:)));
    param_mdl2(:,ii)=qpar{ii,2,idx_best(1)};
end

figure
scatter(param_mdl2(1,:),param_mdl2(2,:),'filled')
xlim([0,1])
ylim([0,1])
title('learning rates')
xlabel('alpha congruent')
ylabel('alpha incongruent')


figure
histogram(param_mdl2(1,:)-param_mdl2(2,:),20)

title('learning rate differences')
xlabel('alpha congruent - alpha incongruent')


figure
scatter(param_mdl2(1,:),param_mdl2(3,:),'filled')
xlim([0,1])
ylim([0,1])
title('alhpa congruent and weight of expectation')
xlabel('alpha congruent')
ylabel('w expectation')



figure
scatter(param_mdl2(2,:),param_mdl2(3,:),'filled')
xlim([0,1])
ylim([0,1])
title('alhpa incongruent and weight of expectation')
xlabel('alpha incongruent')
ylabel('w expectation')


figure
histogram(param_mdl2(3,:),20)
title('histogram of weight of expectation')

% simulating the 63 subjects:
% defining parameters:
for jj=1:size(param_mdl2,2)
    Data=table_pain_new(table_pain_new.src_subject_id==subj_num_new(jj),:);
       idx_ExpH = find(strcmp(Data.param_cue_type, 'high_cue'), 1);
idx_ExpL = find(strcmp(Data.param_cue_type, 'low_cue'), 1);
first_exp_high_cue(jj) = Data.event02_expect_angle(idx_ExpH);
first_exp_low_cue(jj) = Data.event02_expect_angle(idx_ExpL);
        alpha_c=param_mdl2(1,jj);
        alpha_i=param_mdl2(2,jj);
        w=param_mdl2(3,jj);

        % defining expectations and pains:
        num_trial=length(Noxious{jj});
        ExpectationH{jj}=first_exp_high_cue(jj)*ones(num_trial+1,1); % Expectation_highCue matrix
        ExpectationL{jj}=first_exp_low_cue(jj)*ones(num_trial+1,1); % Expectation_lowCue matrix
        Pain{jj}=zeros(num_trial+1,1); % Pain matrix
        PE{jj}=zeros(num_trial+1,1); % Pain predcition error matrix
        PainH{jj}=zeros(num_trial+1,1); % Pain matrix for high cues
        PainL{jj}=zeros(num_trial+1,1); % Pain matrix for low cues
        for ii=1:num_trial
            if High_Cue{jj}(ii)==1
                Pain{jj}(ii,1)=(1-w)*Noxious{jj}(ii,1)+w*ExpectationH{jj}(ii,1);
                PE{jj}(ii,1)=Pain{jj}(ii,1)-ExpectationH{jj}(ii,1);
                if PE{jj}(ii,1)>=0
                    alpha=alpha_c;
                else
                    alpha=alpha_i;
                end
                ExpectationH{jj}(ii+1,1)=ExpectationH{jj}(ii,1)+alpha*PE{jj}(ii,1);
                ExpectationL{jj}(ii+1,1)=ExpectationL{jj}(ii,1);
                PainH{jj}(ii,1)=Pain{jj}(ii,1);
                PainL{jj}(ii,1)=nan;
                nlls{jj}(ii) = (Pain{jj}(ii,1)-Data.event04_actual_angle(ii))^2+(ExpectationH{jj}(ii,1)-Data.event02_expect_angle(ii))^2;
            else
                Pain{jj}(ii,1)=(1-w)*Noxious{jj}(ii,1)+w*ExpectationL{jj}(ii,1);
                PE{jj}(ii,1)=Pain{jj}(ii,1)-ExpectationL{jj}(ii,1);
                if PE{jj}(ii,1)>=0
                    alpha=alpha_i;
                else
                    alpha=alpha_c;
                end
                ExpectationL{jj}(ii+1,1)=ExpectationL{jj}(ii,1)+alpha*PE{jj}(ii,1);
                ExpectationH{jj}(ii+1,1)=ExpectationH{jj}(ii,1);
                PainL{jj}(ii,1)=Pain{jj}(ii,1);
                PainH{jj}(ii,1)=nan;
                nlls{jj}(ii) = (Pain{jj}(ii,1)-Data.event04_actual_angle(ii))^2+(ExpectationL{jj}(ii,1)-Data.event02_expect_angle(ii))^2;
            end
            
        end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first, I add columns tothe table_pain_new that are based on the pain and
% expectation levels of the models:
Pain_mdl2 = [];
for i = 1:numel(Pain)
    vector = Pain{i};
    vector(end) = []; % remove last element
    mean_Pain(i)=nanmean(vector);
    Pain_mdl2 = [Pain_mdl2;vector];
end
table_pain_new.Pain_mdl2=Pain_mdl2;

PainH_mdl2 = [];
for i = 1:numel(PainH)
    vector = PainH{i};
    vector(end) = []; % remove last element
    mean_PainH(i)=nanmean(vector);
    PainH_mdl2 = [PainH_mdl2;vector];
end
table_pain_new.PainH_mdl2=PainH_mdl2;

PainL_mdl2 = [];
for i = 1:numel(PainL)
    vector = PainL{i};
    vector(end) = []; % remove last element
    mean_PainL(i)=nanmean(vector);
    PainL_mdl2 = [PainL_mdl2;vector];
end
table_pain_new.PainL_mdl2=PainL_mdl2;

expH_mdl2 = [];
for i = 1:numel(ExpectationH)
    vector = ExpectationH{i};
    vector(end) = []; % remove last element
    mean_ExpectationH(i)=nanmean(vector);
    expH_mdl2 = [expH_mdl2;vector];
end
table_pain_new.expH_mdl2=expH_mdl2;

expL_mdl2 = [];
for i = 1:numel(ExpectationL)
    vector = ExpectationL{i};
    vector(end) = []; % remove last element
    mean_ExpectationL(i)=nanmean(vector);
    expL_mdl2 = [expL_mdl2;vector];
end
table_pain_new.expL_mdl2=expL_mdl2;


table_pain_new.exp_mdl2 = NaN(size(table_pain_new,1),1);
table_pain_new.exp_mdl2(strcmp(table_pain_new.param_cue_type,'high_cue')) = table_pain_new.expH_mdl2(strcmp(table_pain_new.param_cue_type,'high_cue'));
table_pain_new.exp_mdl2(strcmp(table_pain_new.param_cue_type,'low_cue')) = table_pain_new.expL_mdl2(strcmp(table_pain_new.param_cue_type,'low_cue'));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % subject 1:
% figure
% kk=1;
% plot(ExpectationH{kk}(1:num_trial,:),'--r')
% hold on
% plot(PainH{kk}(1:num_trial,:),'r')
% hold on
% plot(PainL{kk}(1:num_trial,:),'b')
% hold on
% plot(ExpectationL{kk}(1:num_trial,:),'--b')
% legend('ExpH','PainH','PainL','ExpL')


% mean of the expectations and pain reports for each model (subject)
figure
scatter(mean_ExpectationH,mean_PainH,'filled','r')
hold on
scatter(mean_ExpectationL,mean_PainL,'filled','b')
hold on
plot([0, 180], [0, 180])
xlim([0,180])
ylim([0,180])
legend({'High Cue', 'Low Cue'}) 
title('pain rating and expectation ratings in high and low cue')
xlabel('Mean Expectation')
ylabel('Mean Pain Outcome')


figure
histogram(mean_PainH-mean_PainL,20)
xline(median(mean_PainH-mean_PainL),'r')
xline(mean(mean_PainH-mean_PainL),'--r')
title('histogram of difference between PainH and PainL')
[h,p]=ttest(mean_PainH-mean_PainL);
p_med=signrank(mean_PainH-mean_PainL);
subtitle(['mean = ' num2str(mean(mean_PainH-mean_PainL)) ' & pvalue = ' num2str(p) 'median = ' num2str(median(mean_PainH-mean_PainL)) ' & pvalue = ' num2str(p_med)])


figure
histogram(mean_ExpectationH-mean_ExpectationL,20)
xline(median(mean_ExpectationH-mean_ExpectationL),'r')
xline(mean(mean_ExpectationH-mean_ExpectationL),'--r')
title('histogram of difference between ExpectationH and ExpectationL')
[h,p]=ttest(mean_ExpectationH-mean_ExpectationL);
p_med=signrank(mean_ExpectationH-mean_ExpectationL);
subtitle(['mean = ' num2str(mean(mean_ExpectationH-mean_ExpectationL)) ' & pvalue = ' num2str(p) 'median = ' num2str(median(mean_ExpectationH-mean_ExpectationL)) ' & pvalue = ' num2str(p_med)])


%%%%% NOW looking at the behaiovr of the model in three different stimulus
%%%%% intensities:
% in three different stimulus intensitites:
% create a figure with three subplots
stim_type={'low_stim','med_stim','high_stim'};
figure;
for i = 1:3
    % filter the data to include only rows with the current stimulus type
    subset = table_pain_new(strcmp(table_pain_new.param_stimulus_type, stim_type{i}), :);
    % calculate the averages for each ID in each high/low cue
    averages = grpstats(subset, {'src_subject_id', 'param_cue_type'}, {'mean'}, 'DataVars', {'Pain_mdl2', 'exp_mdl2'});
    % create a scatter plot with different colors for high and low cue
    for ii = 1:size(averages, 1)
    if strcmp(averages.param_cue_type{ii}, 'high_cue')
        colors(ii,:) = [1 0 0]; % set red color for high cue
    else
       colors(ii,:) = [0 0 1]; % blue
    end
    end
    subplot(1, 3, i);
    scatter(averages.mean_exp_mdl2,averages.mean_Pain_mdl2, [], colors, 'filled');
    title(['Stimulus Type ', num2str(i)]);
    xlabel('Mean Expectation of model');
    ylabel('Mean Pain of model');
    
end




% On demean data:
% in three different stimulus intensitites:
% create a figure with three subplots
stim_type={'low_stim','med_stim','high_stim'};
meanPain_subj_mdl2 = groupsummary(table_pain_new,'src_subject_id','mean','Pain_mdl2');
meanexp_subj_mdl2 = groupsummary(table_pain_new,'src_subject_id','mean','exp_mdl2');
table_pain_new.paindmean=zeros(size(table_pain_new,1),1);
table_pain_new.expdmean=zeros(size(table_pain_new,1),1);
for ii=1:size(subj_num_new,1)
    idx_subj=table_pain_new.src_subject_id==subj_num_new(ii);
table_pain_new.paindmean_mdl2(idx_subj)=table_pain_new.Pain_mdl2(idx_subj)-meanPain_subj_mdl2.mean_Pain_mdl2(ii);
table_pain_new.expectdmean_mdl2(idx_subj)=table_pain_new.exp_mdl2(idx_subj)-meanexp_subj_mdl2.mean_exp_mdl2(ii);
end

figure;
for i = 1:3
    % filter the data to include only rows with the current stimulus type
    subset = table_pain_new(strcmp(table_pain_new.param_stimulus_type, stim_type{i}), :);
    % calculate the averages for each ID in each high/low cue
    averages = grpstats(subset, {'src_subject_id', 'param_cue_type'}, {'mean'}, 'DataVars', {'expectdmean_mdl2', 'paindmean_mdl2'});
    dmeanpain_mdl2(:,i,1)=averages.mean_paindmean_mdl2(strcmp(averages.param_cue_type, 'low_cue')); % low cue
    dmeanpain_mdl2(:,i,2)=averages.mean_paindmean_mdl2(strcmp(averages.param_cue_type, 'high_cue')); % high cue
    dmeanexp_mdl2(:,i,1)=averages.mean_expectdmean_mdl2(strcmp(averages.param_cue_type, 'low_cue')); % low cue
    dmeanexp_mdl2(:,i,2)=averages.mean_expectdmean_mdl2(strcmp(averages.param_cue_type, 'high_cue')); % high cue
    
    % create a scatter plot with different colors for high and low cue
    for ii = 1:size(averages, 1)
    if strcmp(averages.param_cue_type{ii}, 'high_cue')
        colors(ii,:) = [1 0 0]; % set red color for high cue
    else
       colors(ii,:) = [0 0 1]; % blue
    end
    end
    subplot(1, 3, i);
    scatter(averages.mean_expectdmean_mdl2,averages.mean_paindmean_mdl2, [], colors, 'filled');
    title(['Stimulus Type ', num2str(i)]);
    [r_ILC,p_ILC]=corr(dmeanexp_mdl2(:,i,1),dmeanpain_mdl2(:,i,1),'type','Spearman');
[r_IHC,p_IHC]=corr(dmeanexp_mdl2(:,i,2),dmeanpain_mdl2(:,i,2),'type','Spearman');
subtitle_text = sprintf('corrL=%d & corrH=%d\npvalL=%d & pvalH=%d',r_ILC ,r_IHC ,p_ILC ,p_IHC);
subtitle(subtitle_text)    
    xlabel('demean Expectation model 2');
    ylabel('demean Pain model 2');
    ylim([-60,50])
    xlim([-60,50])
    
end





%% behavioral analysis fo the models that I have fitted:

% calculate the average of pain and expectation for each ID in each high/low cue
averages = grpstats(table_pain_new, {'src_subject_id', 'param_cue_type'}, {'mean'}, 'DataVars', {'event04_actual_angle', 'event02_expect_angle'});
% scatter plot with different colors for high and low cue
for i = 1:size(averages, 1)
    if strcmp(averages.param_cue_type{i}, 'high_cue')
        colors(i,:) = [1 0 0]; % set red color for high cue
    else
       colors(i,:) = [0 0 1]; % blue
    end
end
figure
scatter(averages.mean_event02_expect_angle,averages.mean_event04_actual_angle, [], colors, 'filled');
ylabel('Mean Pain');
xlabel('Mean Expectation');
title('behavioral results');





% dmean:
% calculate mean pain for each participant
meanPain_subj = groupsummary(table_pain_new,'src_subject_id','mean','event04_actual_angle');
meanexp_subj = groupsummary(table_pain_new,'src_subject_id','mean','event02_expect_angle');
table_pain_new.paindmean=zeros(size(table_pain_new,1),1);
table_pain_new.expdmean=zeros(size(table_pain_new,1),1);
for ii=1:size(subj_num_new,1)
    idx_subj=table_pain_new.src_subject_id==subj_num_new(ii);
table_pain_new.paindmean(idx_subj)=table_pain_new.event04_actual_angle(idx_subj)-meanPain_subj.mean_event04_actual_angle(ii);
table_pain_new.expectdmean(idx_subj)=table_pain_new.event02_expect_angle(idx_subj)-meanexp_subj.mean_event02_expect_angle(ii);
end
averages = grpstats(table_pain_new, {'src_subject_id', 'param_cue_type'}, {'mean'}, 'DataVars', {'expectdmean', 'paindmean'});
% scatter plot with different colors for high and low cue
for i = 1:size(averages, 1)
    if strcmp(averages.param_cue_type{i}, 'high_cue')
        colors(i,:) = [1 0 0]; % set red color for high cue
    else
       colors(i,:) = [0 0 1]; % blue
    end
end
figure
scatter(averages.mean_expectdmean,averages.mean_paindmean, [], colors, 'filled');
ylabel('demean Pain');
xlabel('demean Expectation');
title('mean centered behavioral results');


% in three different stimulus intensitites:
% create a figure with three subplots
stim_type={'low_stim','med_stim','high_stim'};
figure;
for i = 1:3
    % filter the data to include only rows with the current stimulus type
    subset = table_pain_new(strcmp(table_pain_new.param_stimulus_type, stim_type{i}), :);
    % calculate the averages for each ID in each high/low cue
    averages = grpstats(subset, {'src_subject_id', 'param_cue_type'}, {'mean'}, 'DataVars', {'event04_actual_angle', 'event02_expect_angle'});
    % create a scatter plot with different colors for high and low cue
    for ii = 1:size(averages, 1)
    if strcmp(averages.param_cue_type{ii}, 'high_cue')
        colors(ii,:) = [1 0 0]; % set red color for high cue
    else
       colors(ii,:) = [0 0 1]; % blue
    end
    end
    subplot(1, 3, i);
    scatter(averages.mean_event02_expect_angle,averages.mean_event04_actual_angle, [], colors, 'filled');
    title(['Stimulus Type ', num2str(i)]);
    xlabel('Mean Expectation');
    ylabel('Mean Pain');
    
end





% On demean data:
% in three different stimulus intensitites:
% create a figure with three subplots
stim_type={'low_stim','med_stim','high_stim'};
figure;
for i = 1:3
    % filter the data to include only rows with the current stimulus type
    subset = table_pain_new(strcmp(table_pain_new.param_stimulus_type, stim_type{i}), :);
    % calculate the averages for each ID in each high/low cue
    averages = grpstats(subset, {'src_subject_id', 'param_cue_type'}, {'mean'}, 'DataVars', {'expectdmean', 'paindmean'});
    dmeanpain_behav(:,i,1)=averages.mean_paindmean(strcmp(averages.param_cue_type, 'low_cue')); % low cue
    dmeanpain_behav(:,i,2)=averages.mean_paindmean(strcmp(averages.param_cue_type, 'high_cue')); % high cue
    dmeanexp_behav(:,i,1)=averages.mean_expectdmean(strcmp(averages.param_cue_type, 'low_cue')); % low cue
    dmeanexp_behav(:,i,2)=averages.mean_expectdmean(strcmp(averages.param_cue_type, 'high_cue')); % high cue
    % create a scatter plot with different colors for high and low cue
    for ii = 1:size(averages, 1)
    if strcmp(averages.param_cue_type{ii}, 'high_cue')
        colors(ii,:) = [1 0 0]; % set red color for high cue
    else
       colors(ii,:) = [0 0 1]; % blue
    end
    end
    subplot(1, 3, i);
    scatter(averages.mean_expectdmean,averages.mean_paindmean, [], colors, 'filled');
    title(['Stimulus Type ', num2str(i)]);
    [r_ILC,p_ILC]=corr(dmeanexp_behav(:,i,1),dmeanpain_behav(:,i,1),'type','Spearman');
[r_IHC,p_IHC]=corr(dmeanexp_behav(:,i,2),dmeanpain_behav(:,i,2),'type','Spearman');
subtitle_text = sprintf('corrL=%d & corrH=%d\npvalL=%d & pvalH=%d',r_ILC ,r_IHC ,p_ILC ,p_IHC);
subtitle(subtitle_text)    
xlabel('demean Expectation');
    ylabel('demean Pain');
    ylim([-60,50])
    xlim([-60,50])
    
end
[r_LILC,p_LILC]=corr(dmeanexp_behav(:,1,1),dmeanpain_behav(:,1,1),'type','Spearman');
[r_MILC,p_MILC]=corr(dmeanexp_behav(:,2,1),dmeanpain_behav(:,2,1),'type','Spearman');
[r_HILC,p_HILC]=corr(dmeanexp_behav(:,3,1),dmeanpain_behav(:,3,1),'type','Spearman');
[r_LIHC,p_LIHC]=corr(dmeanexp_behav(:,1,2),dmeanpain_behav(:,1,2),'type','Spearman');
[r_MIHC,p_MIHC]=corr(dmeanexp_behav(:,2,2),dmeanpain_behav(:,2,2),'type','Spearman');
[r_HIHC,p_HIHC]=corr(dmeanexp_behav(:,3,2),dmeanpain_behav(:,3,2),'type','Spearman');

%% using the demeaned  data to plot pain/expectation behavioral rating versus model rating:
figure
couunt=1;
cue_title={'low cue','high cue'};
noxious_title={'N1','N2','N3'};
for jj=2:-1:1
    for ii=1:3
      subplot(2,3,couunt)
      scatter(dmeanexp_behav(:,ii,jj),dmeanexp_mdl2(:,ii,jj),'filled','k')
      hold on
      plot([-50,50],[-50,50])
      xlabel('behavioral expectation rating')
      ylabel('model expectation rating')
      xlim([-50,50])
      ylim([-50,50])
      title(cue_title{jj})
      subtitle(noxious_title{ii})
      couunt=couunt+1;
      sgtitle('demeaned expectation rating in the cognitive task: model verus behavior')
    end
end


figure
couunt=1;
cue_title={'low cue','high cue'};
noxious_title={'N1','N2','N3'};
for jj=2:-1:1
    for ii=1:3
      subplot(2,3,couunt)
      scatter(dmeanpain_behav(:,ii,jj),dmeanpain_mdl2(:,ii,jj),'filled','k')
      hold on
      plot([-50,50],[-50,50])
      xlabel('behavioral pain rating')
      ylabel('model pain rating')
      xlim([-50,50])
      ylim([-50,50])
      title(cue_title{jj})
      subtitle(noxious_title{ii})
      couunt=couunt+1;
      sgtitle('demeaned pain rating in the cognitive task: model verus behavior')
    end
end


%% Behavioral: learning curves of each individual for pain and expectations:
% for ii=1:size(subj_num_new,1)
for ii=1:10
    Data=table_pain_new(table_pain_new.src_subject_id==subj_num_new(ii),:);

PainLowCue = NaN(size(Data.param_cue_type));
PainLowCue(strcmp(Data.param_cue_type, 'low_cue')) = Data.event04_actual_angle(strcmp(Data.param_cue_type, 'low_cue'));
PainHighCue = NaN(size(Data.param_cue_type));
PainHighCue(strcmp(Data.param_cue_type, 'high_cue')) = Data.event04_actual_angle(strcmp(Data.param_cue_type, 'high_cue'));
ExpLowCue = NaN(size(Data.param_cue_type));
ExpLowCue(strcmp(Data.param_cue_type, 'low_cue')) = Data.event02_expect_angle(strcmp(Data.param_cue_type, 'low_cue'));
ExpHighCue = NaN(size(Data.param_cue_type));
ExpHighCue(strcmp(Data.param_cue_type, 'high_cue')) = Data.event02_expect_angle(strcmp(Data.param_cue_type, 'high_cue'));


% figure % scatter plot for learning pain H and L
% scatter([1:size(Data,1)], PainLowCue, [], 'filled','blue')
% hold on
% scatter([1:size(Data,1)], PainHighCue, [], 'filled','red')
% xlabel('trial number')
% ylabel('Pain outcome')
% legend('low cue', 'high cue')

    figure % plot for learning expectations H and L
    scatter([1:size(Data,1)], ExpLowCue, [], 'filled','blue')
hold on
scatter([1:size(Data,1)], ExpHighCue, [], 'filled','red')
xlabel('trial number')
ylabel('Exp rating')
legend('low cue', 'high cue')
end


%% Behavioral: look at the distribution of the expectation ratings and pain ratings
% as well as the standard deviation of the pain rating an expectation
% ratings:

% for all subjects:
% histograms
figure
histogram(table_pain_new.event02_expect_angle) % hist of expectation
title(['histogram of expectation ratings & std of exp: ' num2str(std(table_pain_new.event02_expect_angle))])
figure
histogram(table_pain_new.event04_actual_angle) % hist of pain
title(['histogram of pain ratings & std of pain: ' num2str(std(table_pain_new.event04_actual_angle))])



% for each of the subjects:
% for ii=1:length(subj_num_new)
for ii=1:10
    sub_no=subj_num_new(ii);
Data=table_pain_new(table_pain_new.src_subject_id==sub_no,:);
% histograms
figure
histogram(Data.event02_expect_angle) % hist of expectation
title(['histogram of expectation ratings & std of exp: ' num2str(std(Data.event02_expect_angle))])
figure
histogram(Data.event04_actual_angle) % hist of pain
title(['histogram of pain ratings & std of pain: ' num2str(std(Data.event04_actual_angle))])
end
%% Hierarchical Beyesian fitting:
clear
close all
clc

chain_numbers=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%% Excluding particpants with few trials or with less than 3 sessions:
PATH_long=[cd '\long_frmt_R'];
table_pain=readtable([PATH_long '\spacetop_cue_pain_0405.csv']);
subj_num=unique(table_pain.src_subject_id);
for ii=1:length(subj_num)
    session_id{ii}=unique(table_pain.session_id(table_pain.src_subject_id==subj_num(ii)));
    num_trial_subj(ii)=max(table_pain.trial_index_subjectwise(table_pain.src_subject_id==subj_num(ii)));
    num_session_subj(ii)=length(session_id{ii});
end
% min(num_trial_subj)
% max(num_trial_subj)
subj_num_new=subj_num;
subj_num_new(num_trial_subj<48 | num_session_subj<3)=[]; % excluding the subjects that have less than 48 trials or less than 3 sessions
selected_rows = ismember(table_pain.src_subject_id, subj_num_new);
table_pain_new=table_pain(selected_rows,:); % creating the new table that exclude subjects with less than 50 trials

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Now I fit the model:
% defining the parameters:
% mdl1) (2(painerror and experror)+2 subject parameters & 6 group parameters) 'mdl1': (1) one alpha (2) one constant weight: w 

% mdl2) (2(painerror and experror)+3 subject parameters & 6 group parameters) 'mdl2 (confirmation bias with one weight)': (1) alpha_c
% (2) alpha_i (3) constant weight: w

% mdl3)(2(painerror and experror)+2 subject parameters & 6 group parameters) 'mdl3 (one alpha with changing weight)':  (1) one alpha
% (2) one changing weight: w=2/(1+exp(gamma*|N-E|)). So parameters are
% alpha and gamma:

% mdl4)(2(painerror and experror)+3 subject parameters & 6 group parameters) 'mdl4 (confirmation bias with one changing weight)': (1)
% (1) alpha_c (2) alpha_i (3) one changing weight: w=2/(1+exp(gamma*|N-E|)). So parameters are
% alpha_c and alpha_i and gamma

model_type={'mdl1', 'mdl2', 'mdl3', 'mdl4'};

qpar=cell(length(model_type),chain_numbers);
trace_cell=cell(length(model_type),chain_numbers);


%%%%%%%%%%%%% should change
initial = 0.5*ones(1,4*63+6);
%%%%%%%%%%%%%%%%%%%%%%%
nsamples = 1000;


        m=1;
        h=1e-6;
%         for model_type={'mdl1', 'mdl2', 'mdl3', 'mdl4'}
        for model_type={'mdl1'}
            tic
            postLL=@(xpar)PosteriorLL_grad(xpar, table_pain_new, model_type{1},subj_num_new);
            
            for jj=1:chain_numbers
%                 [qpar{ii,m,jj}, nll(ii,m,jj), bic(ii,m,jj), aic(ii,m,jj)]=fit_fun_nll(Data,fit_fun,initpar,lb{m},ub{m});

% trace = slicesample(initial,nsamples,'logpdf',postLL);
samples=hmcSampler(postLL,initial);
            end
            m=m+1;
            toc
        end


% save pain_model_fitting_bayesian

%% looking at the nlls in each trial:
clear
close all
clc
load('cognitive_model_fitting_NLL.mat');

%%%%%% finding the best parameters for each subject in each model based on the best aic:
% mdl1
for ii=1:size(aic,1)
    idx_best=find(aic(ii,1,:)==min(aic(ii,1,:)));
    param_mdl1(:,ii)=qpar{ii,1,idx_best(1)};
end
param{1}=param_mdl1;
% mdl2
for ii=1:size(aic,1)
    idx_best=find(aic(ii,2,:)==min(aic(ii,2,:)));
    param_mdl2(:,ii)=qpar{ii,2,idx_best(1)};
end
param{2}=param_mdl2;
% mdl3
for ii=1:size(aic,1)
    idx_best=find(aic(ii,3,:)==min(aic(ii,3,:)));
    param_mdl3(:,ii)=qpar{ii,3,idx_best(1)};
end
param{3}=param_mdl3;
% mdl4
for ii=1:size(aic,1)
    idx_best=find(aic(ii,4,:)==min(aic(ii,4,:)));
    param_mdl4(:,ii)=qpar{ii,4,idx_best(1)};
end
param{4}=param_mdl4;
%%%%%%

nll_mdl1_trialwise=[];
nll_mdl2_trialwise=[];
nll_mdl3_trialwise=[];
nll_mdl4_trialwise=[];
Pain_mdl1_trialwise=[];
Pain_mdl2_trialwise=[];
Pain_mdl3_trialwise=[];
Pain_mdl4_trialwise=[];
Exp_mdl1_trialwise=[];
Exp_mdl2_trialwise=[];
Exp_mdl3_trialwise=[];
Exp_mdl4_trialwise=[];
PE_mdl1_trialwise=[];
PE_mdl2_trialwise=[];
PE_mdl3_trialwise=[];
PE_mdl4_trialwise=[];

% for each subject and each model:
qpar=cell(length(subj_num_new),length(model_type),run_numbers);
for ii=1:length(subj_num_new)
ii
sub_no=subj_num_new(ii);
% 0) get the data matrix for the subject:
Data=table_pain_new(table_pain_new.src_subject_id==sub_no,:);
Data_lowpain=Data(strcmp(Data.param_stimulus_type, 'low_stim'),:);
Data_medpain=Data(strcmp(Data.param_stimulus_type, 'med_stim'),:);
Data_highpain=Data(strcmp(Data.param_stimulus_type, 'high_stim'),:);
% 1)	Calculate the Noxious inputs (1, 2, or 3) for each of the participants based on the average ratings of that participant on each of those real Noxious inputs. For example, to calculate the subjective Noxious 1 (48) for participant 1, we average the ratings on the trials that the participant has got 48-degree heat (and the pain rating is reported on 0-180 scale). 
% From pain rating vector: 
N(ii,1)=nanmean(Data_lowpain.event04_actual_angle);
N(ii,2)=nanmean(Data_medpain.event04_actual_angle);
N(ii,3)=nanmean(Data_highpain.event04_actual_angle);
% Noxious input vecotr for each participant:
Noxious{ii} = zeros(length(Data.param_stimulus_type),1);
Noxious{ii}(strcmp(Data.param_stimulus_type,'low_stim')) = N(ii,1);
Noxious{ii}(strcmp(Data.param_stimulus_type,'med_stim')) = N(ii,2);
Noxious{ii}(strcmp(Data.param_stimulus_type,'high_stim')) = N(ii,3);
% High_Cue vector for each participant:
High_Cue{ii} = strcmp(Data.param_cue_type,'high_cue');
% 2, 3, 4)	Calculate the Expectation and Pain matrices based on each model
% Calculate the nll based on the above equation
% For jj=1:20 % run 20 times to avoid being stuck in the 
%   Optimize the nll function and get the parameters and nll for each model and pick the best parameter
% End
% 5)	Calculate aic, bic, and nll or each model and compare different models together
Noxious_subj=Noxious{ii};
High_Cue_subj=High_Cue{ii};
[nlls1,Pain1,Expectation1,PE1,nlls_pain1,nlls_exp1]=NLL_trialwise(param{1}(:,ii), Data, 'mdl1',Noxious_subj,High_Cue_subj);
[nlls2,Pain2,Expectation2,PE2,nlls_pain2,nlls_exp2]=NLL_trialwise(param{2}(:,ii), Data, 'mdl2',Noxious_subj,High_Cue_subj);
[nlls3,Pain3,Expectation3,PE3,nlls_pain3,nlls_exp3]=NLL_trialwise(param{3}(:,ii), Data, 'mdl3',Noxious_subj,High_Cue_subj);
[nlls4,Pain4,Expectation4,PE4,nlls_pain4,nlls_exp4]=NLL_trialwise(param{4}(:,ii), Data, 'mdl4',Noxious_subj,High_Cue_subj);

            nll_mdl1_trialwise=[nll_mdl1_trialwise;nlls1'];
            nll_mdl2_trialwise=[nll_mdl2_trialwise;nlls2'];
            nll_mdl3_trialwise=[nll_mdl3_trialwise;nlls3'];
            nll_mdl4_trialwise=[nll_mdl4_trialwise;nlls4'];

            Pain_mdl1_trialwise=[Pain_mdl1_trialwise;Pain1];
            Pain_mdl2_trialwise=[Pain_mdl2_trialwise;Pain2];
            Pain_mdl3_trialwise=[Pain_mdl3_trialwise;Pain3];
            Pain_mdl4_trialwise=[Pain_mdl4_trialwise;Pain4];

            Exp_mdl1_trialwise=[Exp_mdl1_trialwise;Expectation1];
            Exp_mdl2_trialwise=[Exp_mdl2_trialwise;Expectation2];
            Exp_mdl3_trialwise=[Exp_mdl3_trialwise;Expectation3];
            Exp_mdl4_trialwise=[Exp_mdl4_trialwise;Expectation4];

            PE_mdl1_trialwise=[PE_mdl1_trialwise;PE1];
            PE_mdl2_trialwise=[PE_mdl2_trialwise;PE2];
            PE_mdl3_trialwise=[PE_mdl3_trialwise;PE3];
            PE_mdl4_trialwise=[PE_mdl4_trialwise;PE4];
            painnll1(ii)=nlls_pain1;
            expnll1(ii)=nlls_exp1;
            painnll2(ii)=nlls_pain2;
            expnll2(ii)=nlls_exp2;
            painnll3(ii)=nlls_pain3;
            expnll3(ii)=nlls_exp3;
            painnll4(ii)=nlls_pain4;
            expnll4(ii)=nlls_exp4;

      
end


table_pain_new.nll_mdl1=nll_mdl1_trialwise;
table_pain_new.nll_mdl2=nll_mdl2_trialwise;
table_pain_new.nll_mdl3=nll_mdl3_trialwise;
table_pain_new.nll_mdl4=nll_mdl4_trialwise;

table_pain_new.PE_mdl1=PE_mdl1_trialwise;
table_pain_new.PE_mdl2=PE_mdl2_trialwise;
table_pain_new.PE_mdl3=PE_mdl3_trialwise;
table_pain_new.PE_mdl4=PE_mdl4_trialwise;

table_pain_new.Pain_mdl1=Pain_mdl1_trialwise;
table_pain_new.Pain_mdl2=Pain_mdl2_trialwise;
table_pain_new.Pain_mdl3=Pain_mdl3_trialwise;
table_pain_new.Pain_mdl4=Pain_mdl4_trialwise;

table_pain_new.Exp_mdl1=Exp_mdl1_trialwise;
table_pain_new.Exp_mdl2=Exp_mdl2_trialwise;
table_pain_new.Exp_mdl3=Exp_mdl3_trialwise;
table_pain_new.Exp_mdl4=Exp_mdl4_trialwise;


%%%%%% looking at trialwise nll for each subject:
% for ii=1:length(subj_num_new)
for ii=1:20
sub_no=subj_num_new(ii);
Data=table_pain_new(table_pain_new.src_subject_id==sub_no,:);
time=1:size(Data,1);
idx = find(diff(Data.session_id) ~= 0);
figure
plot(time,Data.nll_mdl2)
hold on
plot(time,Data.nll_mdl4,'r')
xline(time(idx+1), 'k--')
legend('mdl2','mdl4')
end


% for ii=1:length(subj_num_new) % diffbetween mdl4 and mdl2
for ii=1:20
sub_no=subj_num_new(ii);
Data=table_pain_new(table_pain_new.src_subject_id==sub_no,:);
time=1:size(Data,1);
idx = find(diff(Data.session_id) ~= 0);
figure
plot(time,Data.nll_mdl4-Data.nll_mdl2)
xline(time(idx+1), 'k--')
end

% violin plots of the nlls in 6 conditions:
% Get the indices for each combination
idx_low_cue_low_stim = table_pain_new.param_cue_type == "low_cue" & table_pain_new.param_stimulus_type == "low_stim";
idx_low_cue_med_stim = table_pain_new.param_cue_type == "low_cue" & table_pain_new.param_stimulus_type == "med_stim";
idx_low_cue_high_stim = table_pain_new.param_cue_type == "low_cue" & table_pain_new.param_stimulus_type == "high_stim";
idx_high_cue_low_stim = table_pain_new.param_cue_type == "high_cue" & table_pain_new.param_stimulus_type == "low_stim";
idx_high_cue_med_stim = table_pain_new.param_cue_type == "high_cue" & table_pain_new.param_stimulus_type == "med_stim";
idx_high_cue_high_stim = table_pain_new.param_cue_type == "high_cue" & table_pain_new.param_stimulus_type == "high_stim";
% 1) low intensity low cue
dat1 = table_pain_new.nll_mdl2(idx_low_cue_low_stim);
% 2) low intensity high cue
dat2 = table_pain_new.nll_mdl2(idx_high_cue_low_stim);
% 3) med intensity low cue
dat3 = table_pain_new.nll_mdl2(idx_low_cue_med_stim);
% 4) med intensity high cue
dat4 = table_pain_new.nll_mdl2(idx_high_cue_med_stim);
% 5) high intensity low cue
dat5 = table_pain_new.nll_mdl2(idx_low_cue_high_stim);
% 6) high intrnsity high cue
dat6 = table_pain_new.nll_mdl2(idx_high_cue_high_stim);
% beta mag

Data = [dat1; dat2; dat3;dat4; dat5; dat6]; 
Labels = [repelem([{'Low In, Low Cue'}, {'Low In, High Cue'}, {'Med In, Low Cue'}, {'Med In, High Cue'}, {'High In, Low Cue'}, {'High In, High Cue'}], ...
    [length(dat1), length(dat2), length(dat3),length(dat4), length(dat5), length(dat6)])]';
grouporder = {'Low In, Low Cue', 'Low In, High Cue', 'Med In, Low Cue','Med In, High Cue', 'High In, Low Cue', 'High In, High Cue'};
figure
ax = gca;
% vs = violinplotV2(Data, Labels, [1 : 6],'GroupOrder',grouporder, 'ShowMean', true, 'MedianColor', [0,0,0]);
vs = violinplotV2(Data, Labels, [1 : 6],'GroupOrder',grouporder);
box off


figure
boxplt = boxplot(Data, Labels);
set(boxplt, 'LineWidth', 1.5);
title('nlls of model 2')


% nll_pain and nll_exp for subjects:
% mdl1
figure
Labels_pain_exp=[repelem([{'pain nll'}, {'exp nll'}], ...
    [length(painnll1), length(expnll1)])]';
data_pain_exp=[painnll1,expnll1]';
boxplt = boxplot(data_pain_exp, Labels_pain_exp);
set(boxplt, 'LineWidth', 1.5);
title('nlls of model 1')
% mdl2
figure
Labels_pain_exp=[repelem([{'pain nll'}, {'exp nll'}], ...
    [length(painnll2), length(expnll2)])]';
data_pain_exp=[painnll2,expnll2]';
boxplt = boxplot(data_pain_exp, Labels_pain_exp);
set(boxplt, 'LineWidth', 1.5);
title('nlls of model 2')
% mdl3
figure
Labels_pain_exp=[repelem([{'pain nll'}, {'exp nll'}], ...
    [length(painnll3), length(expnll3)])]';
data_pain_exp=[painnll3,expnll3]';
boxplt = boxplot(data_pain_exp, Labels_pain_exp);
set(boxplt, 'LineWidth', 1.5);
title('nlls of model 3')
% mdl4
figure
Labels_pain_exp=[repelem([{'pain nll'}, {'exp nll'}], ...
    [length(painnll4), length(expnll4)])]';
data_pain_exp=[painnll4,expnll4]';
boxplt = boxplot(data_pain_exp, Labels_pain_exp);
set(boxplt, 'LineWidth', 1.5);
title('nlls of model 4')

%% correlation between the exp and pain in each of the 6 combinations:

% behavioral results:

% model 2 results:
[r_LILC_m2,p_LILC_m2]=corr(dmeanexp_mdl2(:,1,1),dmeanpain_mdl2(:,1,1),'type','Spearman');
[r_MILC_m2,p_MILC_m2]=corr(dmeanexp_mdl2(:,2,1),dmeanpain_mdl2(:,2,1),'type','Spearman');
[r_HILC_m2,p_HILC_m2]=corr(dmeanexp_mdl2(:,3,1),dmeanpain_mdl2(:,3,1),'type','Spearman');
[r_LIHC_m2,p_LIHC_m2]=corr(dmeanexp_mdl2(:,1,2),dmeanpain_mdl2(:,1,2),'type','Spearman');
[r_MIHC_m2,p_MIHC_m2]=corr(dmeanexp_mdl2(:,2,2),dmeanpain_mdl2(:,2,2),'type','Spearman');
[r_HIHC_m2,p_HIHC_m2]=corr(dmeanexp_mdl2(:,3,2),dmeanpain_mdl2(:,3,2),'type','Spearman');

%% look at the variance of pain and exp rating for each subject:
% load('cognitive_model_fitting_NLL.mat');
for ii=1:length(subj_num_new)
    idx_ii_L=table_pain_new.param_cue_type=="low_cue" & table_pain_new.src_subject_id==subj_num_new(ii);
    idx_ii_H=table_pain_new.param_cue_type=="high_cue" & table_pain_new.src_subject_id==subj_num_new(ii);
    
    std_painH(ii)=std(table_pain_new.event04_actual_angle(idx_ii_H));
    std_expH(ii)=std(table_pain_new.event02_expect_angle(idx_ii_H));
    std_painL(ii)=std(table_pain_new.event04_actual_angle(idx_ii_L));
    std_expL(ii)=std(table_pain_new.event02_expect_angle(idx_ii_L));
    idx_ii=table_pain_new.src_subject_id==subj_num_new(ii);
    std_pain(ii)=std(table_pain_new.event04_actual_angle(idx_ii));
    std_exp(ii)=std(table_pain_new.event02_expect_angle(idx_ii));
end


figure
scatter(std_expL,std_painL,'filled','k')
[h,p]=ttest2(std_expL,std_painL);
hold on
plot([1:60],[1:60],'r')
ylabel('pain std L')
xlabel('exp std L')
title('standard deviation of subjective reports in Low cue')
subtitle(['ttest, pval = ' num2str(p)])


figure
scatter(std_expH,std_painH,'filled','k')
[h,p]=ttest2(std_expH,std_painH);
hold on
plot([1:60],[1:60],'r')
ylabel('pain std H')
xlabel('exp std H')
title('standard deviation of subjective reports in High cue')
subtitle(['ttest, pval = ' num2str(p)])


figure
scatter(param_mdl2(4,:),std_painL,'filled','k')
figure
scatter(param_mdl2(4,:),std_painH,'filled','k')
figure
scatter(param_mdl2(5,:),std_expL,'filled','k')
figure
scatter(param_mdl2(5,:),std_expH,'filled','k')

figure
scatter(param_mdl2(4,:),std_pain,'filled','k')
xlabel('pain error in the model')
ylabel('std of pain in behavioral reports')
hold on
plot([1:60],[1:60],'r')
xlim([0,60])
ylim([0,60])

figure
scatter(param_mdl2(5,:),std_exp,'filled','k')
xlabel('exp error in the model')
ylabel('std of exp in behavioral reports')
hold on
plot([1:60],[1:60],'r')
xlim([0,60])
ylim([0,60])

% subplots of NLL and std of reports (systematic noise):
figure
% looking at the NLL and the variance of pain in behavioral reports
subplot(2,2,1)
scatter(nll_mat(:,2),std_pain,'filled')
[r,p]=corr(nll_mat(:,2),std_pain','Type','Spearman');
subtitle(['corr =  ' num2str(r) ' & pval = ' num2str(p)])
title('pain behavioral')
xlabel('NLL')
ylabel('std pain behavioral')
% looking at the NLL and the pain error in the model
subplot(2,2,2)
scatter(nll_mat(:,2),param_mdl2(4,:),'filled')
[r,p]=corr(nll_mat(:,2),param_mdl2(4,:)','Type','Spearman');
subtitle(['corr =  ' num2str(r) ' & pval = ' num2str(p)])
title('pain model')
xlabel('NLL')
ylabel('error of pain in model')
% looking at the NLL and the variance of exp in behavioral reports
subplot(2,2,3)
scatter(nll_mat(:,2),std_exp,'filled')
[r,p]=corr(nll_mat(:,2),std_exp','Type','Spearman');
subtitle(['corr =  ' num2str(r) ' & pval = ' num2str(p)])
title('exp behavioral')
xlabel('NLL')
ylabel('std exop behavioral')
% looking at the NLL and the exp error in the model
subplot(2,2,4)
scatter(nll_mat(:,2),param_mdl2(5,:),'filled')
[r,p]=corr(nll_mat(:,2),param_mdl2(5,:)','Type','Spearman');
subtitle(['corr =  ' num2str(r) ' & pval = ' num2str(p)])
title('exp model')
xlabel('NLL')
ylabel('error of exp in model')
sgtitle('Cognitive task')
%% making a table of the parameters
load('param_mdl2_cognitive.mat');
load('subj_num_new_cognitive.mat');
alpha_c=param_mdl2_cognitive(1,:)';
alpha_i=param_mdl2_cognitive(2,:)';
weight_exp=param_mdl2_cognitive(3,:)';
pain_err=param_mdl2_cognitive(4,:)';
exp_err=param_mdl2_cognitive(5,:)';
par_mdl2_cog=table(subj_num_new_cognitive,alpha_c,alpha_i,weight_exp,pain_err,exp_err);
writetable(par_mdl2_cog,'par_mdl2_cog.csv');